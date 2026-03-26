// cmd/server/main.go
//
// HTTP server that exposes POST /verify — the full face verification pipeline:
//  1. Receive a "live" photo from the mobile app (base64 or multipart)
//  2. Cloud Vision: NSFW check + exactly-one-face check
//  3. Pull the stored "profile" photo (mocked as a local file for now)
//  4. Vertex AI: extract embeddings for both images
//  5. Cosine similarity → verified / rejected
//     Returns JSON: { "verified": bool, "similarity": float, "message": string }
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"

	vision "cloud.google.com/go/vision/apiv1"
	visionpb "cloud.google.com/go/vision/v2/apiv1/visionpb"
	"github.com/joho/godotenv"
	"golang.org/x/oauth2/google"
)

// ── Configuration ──────────────────────────────────────────────────────

const (
	similarityThreshold = 0.5
	serverPort          = ":8080"
	// For the PoC, the "stored" profile photo is a local file.
	// In production this would come from a database / cloud storage.
	mockProfilePhotoPath = "test_profile_photo.jpg"
)

// ── Vertex AI types ────────────────────────────────────────────────────

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

// ── API response ───────────────────────────────────────────────────────

type VerifyResponse struct {
	Verified   bool    `json:"verified"`
	Similarity float64 `json:"similarity"`
	Message    string  `json:"message"`
}

// ── Global clients (initialised once at startup) ───────────────────────

var (
	visionClient *vision.ImageAnnotatorClient
	gcpProjectID string
	gcpLocation  string
)

// ── Main ───────────────────────────────────────────────────────────────

func main() {
	// Load .env
	if err := godotenv.Overload(); err != nil {
		log.Println("No .env file found, relying on environment variables")
	}

	gcpProjectID = os.Getenv("GCP_PROJECT_ID")
	if gcpProjectID == "" {
		log.Fatal("GCP_PROJECT_ID is not set. Add it to .env or export it.")
	}
	gcpLocation = os.Getenv("GCP_LOCATION")
	if gcpLocation == "" {
		gcpLocation = "us-central1"
	}

	// Initialise Cloud Vision client
	ctx := context.Background()
	var err error
	visionClient, err = vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create Vision client: %v", err)
	}
	defer visionClient.Close()

	// Routes
	http.HandleFunc("/verify", handleVerify)
	http.HandleFunc("/health", handleHealth)

	log.Printf("Face verification server listening on %s", serverPort)
	log.Fatal(http.ListenAndServe(serverPort, nil))
}

// ── /health ────────────────────────────────────────────────────────────

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// ── /verify ────────────────────────────────────────────────────────────

func handleVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, http.StatusMethodNotAllowed, "Only POST is accepted")
		return
	}

	// ── 1. Extract the live photo bytes ────────────────────────────────
	livePhotoBytes, err := extractLivePhoto(r)
	if err != nil {
		jsonError(w, http.StatusBadRequest, fmt.Sprintf("Could not read live photo: %v", err))
		return
	}
	log.Printf("Received live photo: %d bytes", len(livePhotoBytes))

	ctx := context.Background()

	// ── 2. Cloud Vision: NSFW + single-face check ─────────────────────
	visionImage, err := vision.NewImageFromReader(bytes.NewReader(livePhotoBytes))
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to create Vision image: %v", err))
		return
	}

	// Face count
	faces, err := visionClient.DetectFaces(ctx, visionImage, nil, 10)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Face detection failed: %v", err))
		return
	}
	if len(faces) != 1 {
		jsonResp(w, VerifyResponse{
			Verified:   false,
			Similarity: 0,
			Message:    fmt.Sprintf("Expected exactly 1 face, found %d", len(faces)),
		})
		return
	}
	log.Println("✓ Exactly one face detected")

	// Safe Search
	safeSearch, err := visionClient.DetectSafeSearch(ctx, visionImage, nil)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Safe search failed: %v", err))
		return
	}
	if safeSearch.Adult >= visionpb.Likelihood_LIKELY || safeSearch.Violence >= visionpb.Likelihood_LIKELY {
		jsonResp(w, VerifyResponse{
			Verified:   false,
			Similarity: 0,
			Message:    "Image flagged as NSFW",
		})
		return
	}
	log.Println("✓ Image passed NSFW check")

	// ── 3. Load the stored profile photo ──────────────────────────────
	profilePhotoBytes, err := os.ReadFile(mockProfilePhotoPath)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Could not load profile photo: %v", err))
		return
	}

	// ── 4. Vertex AI: get embeddings for both images ──────────────────
	liveEmbedding, err := getImageEmbedding(livePhotoBytes)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to get live photo embedding: %v", err))
		return
	}
	log.Printf("✓ Live photo embedding: %d dimensions", len(liveEmbedding))

	profileEmbedding, err := getImageEmbedding(profilePhotoBytes)
	if err != nil {
		jsonError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to get profile photo embedding: %v", err))
		return
	}
	log.Printf("✓ Profile photo embedding: %d dimensions", len(profileEmbedding))

	// ── 5. Cosine similarity ──────────────────────────────────────────
	similarity := cosineSimilarity(liveEmbedding, profileEmbedding)
	verified := similarity >= similarityThreshold

	msg := "REJECTED — faces do not match"
	if verified {
		msg = "VERIFIED — same person"
	}
	log.Printf("Similarity: %.6f — %s", similarity, msg)

	jsonResp(w, VerifyResponse{
		Verified:   verified,
		Similarity: similarity,
		Message:    msg,
	})
}

// ── Image extraction helpers ───────────────────────────────────────────

// extractLivePhoto supports two content types:
//   - application/json  → { "image": "<base64>" }
//   - multipart/form-data → field name "image"
func extractLivePhoto(r *http.Request) ([]byte, error) {
	contentType := r.Header.Get("Content-Type")

	if strings.HasPrefix(contentType, "multipart/form-data") {
		// 32 MB max
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			return nil, fmt.Errorf("parse multipart: %w", err)
		}
		file, _, err := r.FormFile("image")
		if err != nil {
			return nil, fmt.Errorf("get form file 'image': %w", err)
		}
		defer file.Close()
		return io.ReadAll(file)
	}

	// Default: JSON with base64
	var body struct {
		Image string `json:"image"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	return base64.StdEncoding.DecodeString(body.Image)
}

// ── Vertex AI embedding ───────────────────────────────────────────────

func getImageEmbedding(imgBytes []byte) ([]float64, error) {
	b64 := base64.StdEncoding.EncodeToString(imgBytes)

	reqData := PredictRequest{
		Instances: []Instance{{Image: ImagePayload{BytesBase64Encoded: b64}}},
	}
	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("auth client: %w", err)
	}

	url := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/multimodalembedding@001:predict",
		gcpLocation, gcpProjectID, gcpLocation,
	)

	resp, err := client.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("API request: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API status %d: %s", resp.StatusCode, string(body))
	}

	var predictResp PredictResponse
	if err := json.Unmarshal(body, &predictResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	if len(predictResp.Predictions) == 0 {
		return nil, fmt.Errorf("no embeddings returned: %s", string(body))
	}

	return predictResp.Predictions[0].ImageEmbedding, nil
}

// ── Math ───────────────────────────────────────────────────────────────

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ── JSON helpers ───────────────────────────────────────────────────────

func jsonResp(w http.ResponseWriter, v VerifyResponse) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func jsonError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(VerifyResponse{
		Verified: false,
		Message:  msg,
	})
}
