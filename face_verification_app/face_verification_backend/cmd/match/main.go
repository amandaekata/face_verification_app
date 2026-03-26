// cmd/match/main.go
//
// Vertex AI Identity Matcher: extracts multimodal embeddings from two images
// and computes cosine similarity to verify they depict the same person.
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"golang.org/x/oauth2/google"
)

// ── API Request & Response Structures ──────────────────────────────────

type PredictRequest struct {
	Instances []Instance `json:"instances"`
}

type Instance struct {
	Image ImagePayload `json:"image"`
}

type ImagePayload struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
}

type PredictResponse struct {
	Predictions []Prediction `json:"predictions"`
}

type Prediction struct {
	ImageEmbedding []float64 `json:"imageEmbedding"`
}

// ── Core Functions ─────────────────────────────────────────────────────

// getImageEmbedding sends an image to the Vertex AI Multimodal Embedding
// endpoint and returns its vector representation.
func getImageEmbedding(projectID, location, imagePath string) ([]float64, error) {
	// 1. Read and base64-encode the image
	imgData, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read image: %v", err)
	}
	b64Image := base64.StdEncoding.EncodeToString(imgData)

	// 2. Build the JSON request payload
	reqData := PredictRequest{
		Instances: []Instance{{Image: ImagePayload{BytesBase64Encoded: b64Image}}},
	}
	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// 3. Get a Google Cloud auth token via Application Default Credentials
	ctx := context.Background()
	client, err := google.DefaultClient(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("failed to get auth client: %v", err)
	}

	// 4. POST to the Vertex AI Multimodal Embedding model
	url := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/multimodalembedding@001:predict",
		location, projectID, location,
	)

	resp, err := client.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()

	// 5. Parse response
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var predictResp PredictResponse
	if err := json.Unmarshal(body, &predictResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	if len(predictResp.Predictions) == 0 {
		return nil, fmt.Errorf("no embeddings returned. Response: %s", string(body))
	}

	return predictResp.Predictions[0].ImageEmbedding, nil
}

// cosineSimilarity calculates how close two embedding vectors are.
// Returns a value from -1 (opposite) to +1 (identical).
func cosineSimilarity(a, b []float64) float64 {
	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ── Main ───────────────────────────────────────────────────────────────

const similarityThreshold = 0.5

func main() {
	// ── Configuration ──────────────────────────────────────────────────
	projectID := os.Getenv("GCP_PROJECT_ID")
	if projectID == "" {
		fmt.Println("Error: GCP_PROJECT_ID environment variable is not set.")
		fmt.Println("Usage: GCP_PROJECT_ID=your-project-id go run .")
		os.Exit(1)
	}

	location := os.Getenv("GCP_LOCATION")
	if location == "" {
		location = "us-central1"
	}

	profilePhoto := "test_profile_photo.jpg"
	livePhoto := "test_live_photo.jpg"
	if len(os.Args) > 2 {
		profilePhoto = os.Args[1]
		livePhoto = os.Args[2]
	}

	// ── Embedding Extraction ───────────────────────────────────────────
	fmt.Println("── Step 1: Extracting embedding from Profile Photo ──")
	profileVector, err := getImageEmbedding(projectID, location, profilePhoto)
	if err != nil {
		fmt.Printf("  Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  ✓ Got %d-dimensional embedding.\n", len(profileVector))

	fmt.Println("\n── Step 2: Extracting embedding from Live Challenge Photo ──")
	liveVector, err := getImageEmbedding(projectID, location, livePhoto)
	if err != nil {
		fmt.Printf("  Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  ✓ Got %d-dimensional embedding.\n", len(liveVector))

	// ── Comparison ─────────────────────────────────────────────────────
	similarity := cosineSimilarity(profileVector, liveVector)

	fmt.Println("\n══ Verification Results ══")
	fmt.Printf("  Similarity Score: %.6f\n", similarity)
	fmt.Printf("  Threshold:        %.2f\n", similarityThreshold)

	if similarity >= similarityThreshold {
		fmt.Println("  Status: ✓ VERIFIED — These photos belong to the same person!")
	} else {
		fmt.Println("  Status: ✗ REJECTED — These are likely different people.")
		os.Exit(1)
	}
}
