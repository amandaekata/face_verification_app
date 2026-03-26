// cmd/validate/main.go
//
// First line of defense: validates that an image passes the NSFW check
// and contains exactly one face using Google Cloud Vision API.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	vision "cloud.google.com/go/vision/apiv1"
	visionpb "cloud.google.com/go/vision/v2/apiv1/visionpb"
)

func main() {
	ctx := context.Background()

	// Initialize the Cloud Vision client.
	// Requires GOOGLE_APPLICATION_CREDENTIALS env var to be set.
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create Vision client: %v", err)
	}
	defer client.Close()

	// Use a local test image for the PoC.
	// In production this would be the uploaded profile photo.
	filename := "test_profile_photo.jpg"
	if len(os.Args) > 1 {
		filename = os.Args[1]
	}

	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Failed to read file %q: %v", filename, err)
	}
	defer file.Close()

	image, err := vision.NewImageFromReader(file)
	if err != nil {
		log.Fatalf("Failed to create image: %v", err)
	}

	passed := true

	// ── 1. Face Detection ─────────────────────────────────────────────────
	fmt.Println("── Step 1: Face Detection ──")
	faces, err := client.DetectFaces(ctx, image, nil, 10)
	if err != nil {
		log.Fatalf("Failed to detect faces: %v", err)
	}

	if len(faces) != 1 {
		fmt.Printf("  ✗ FAILED — Found %d face(s). Exactly one is required.\n", len(faces))
		passed = false
	} else {
		fmt.Println("  ✓ PASSED — Exactly one face detected.")
	}

	// ── 2. Safe Search (NSFW) ─────────────────────────────────────────────
	fmt.Println("\n── Step 2: Safe Search (NSFW Check) ──")
	safeSearch, err := client.DetectSafeSearch(ctx, image, nil)
	if err != nil {
		log.Fatalf("Failed to detect safe search properties: %v", err)
	}

	fmt.Printf("  Adult:    %s\n", safeSearch.Adult)
	fmt.Printf("  Violence: %s\n", safeSearch.Violence)
	fmt.Printf("  Racy:     %s\n", safeSearch.Racy)

	if safeSearch.Adult >= visionpb.Likelihood_LIKELY || safeSearch.Violence >= visionpb.Likelihood_LIKELY {
		fmt.Println("  ✗ FAILED — Image flagged as NSFW.")
		passed = false
	} else {
		fmt.Println("  ✓ PASSED — Image is safe.")
	}

	// ── Result ────────────────────────────────────────────────────────────
	fmt.Println()
	if passed {
		fmt.Println("══ RESULT: Image PASSED all validation checks. ══")
	} else {
		fmt.Println("══ RESULT: Image FAILED validation. ══")
		os.Exit(1)
	}
}
