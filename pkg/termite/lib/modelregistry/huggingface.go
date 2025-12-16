// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package modelregistry

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
)

// HuggingFaceClient pulls ONNX models from HuggingFace Hub
type HuggingFaceClient struct {
	token           string
	progressHandler ProgressHandler
}

// HFClientOption configures the HuggingFace client
type HFClientOption func(*HuggingFaceClient)

// NewHuggingFaceClient creates a new HuggingFace client
func NewHuggingFaceClient(opts ...HFClientOption) *HuggingFaceClient {
	c := &HuggingFaceClient{}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// WithHFToken sets the HuggingFace API token for gated models
func WithHFToken(token string) HFClientOption {
	return func(c *HuggingFaceClient) { c.token = token }
}

// WithHFProgressHandler sets the progress handler for downloads
func WithHFProgressHandler(h ProgressHandler) HFClientOption {
	return func(c *HuggingFaceClient) { c.progressHandler = h }
}

// PullFromHuggingFace downloads ONNX model files from a HuggingFace repo.
// variant can be: "", "fp16", "q4", "q4f16", "quantized"
func (c *HuggingFaceClient) PullFromHuggingFace(
	ctx context.Context,
	repoID string,
	modelType ModelType,
	destDir string,
	variant string,
) error {
	repo := hub.New(repoID)
	if c.token != "" {
		repo = repo.WithAuth(c.token)
	}

	// List all files in repo
	var files []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return fmt.Errorf("listing files: %w", err)
		}
		files = append(files, fileName)
	}

	// Filter and select files to download
	toDownload := selectONNXFiles(files, variant)
	if len(toDownload) == 0 {
		return fmt.Errorf("no ONNX files found in %s", repoID)
	}

	// Create destination directory
	modelName := filepath.Base(repoID)
	modelDir := filepath.Join(destDir, modelType.DirName(), modelName)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("creating directory: %w", err)
	}

	// Download each file
	for _, fileName := range toDownload {
		localPath, err := repo.DownloadFile(fileName)
		if err != nil {
			return fmt.Errorf("downloading %s: %w", fileName, err)
		}

		// Flatten path (e.g., "onnx/model.onnx" -> "model.onnx")
		destName := filepath.Base(fileName)
		destPath := filepath.Join(modelDir, destName)

		// Report progress before copy
		if c.progressHandler != nil {
			c.progressHandler(0, 0, destName)
		}

		// Copy from cache to destination
		if err := copyFile(localPath, destPath); err != nil {
			return fmt.Errorf("copying %s: %w", fileName, err)
		}

		// Report completion
		if c.progressHandler != nil {
			if info, err := os.Stat(destPath); err == nil {
				c.progressHandler(info.Size(), info.Size(), destName)
			}
		}
	}

	return nil
}

// selectONNXFiles filters files based on variant preference.
// It returns tokenizer files plus the ONNX model file(s) matching the variant.
func selectONNXFiles(files []string, variant string) []string {
	var result []string

	// Always include tokenizer/config files from anywhere in the repo
	tokenizerFiles := []string{"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "config.json", "special_tokens_map.json"}
	for _, tf := range tokenizerFiles {
		for _, f := range files {
			if filepath.Base(f) == tf {
				result = append(result, f)
				break
			}
		}
	}

	// Determine ONNX file pattern based on variant
	var onnxBase string
	switch variant {
	case "fp16":
		onnxBase = "model_fp16"
	case "q4":
		onnxBase = "model_q4"
	case "q4f16":
		onnxBase = "model_q4f16"
	case "quantized":
		onnxBase = "model_quantized"
	default:
		onnxBase = "model"
	}

	// Find matching ONNX files (model.onnx + model.onnx_data)
	for _, f := range files {
		base := filepath.Base(f)
		// Match exact model file or its data file
		if base == onnxBase+".onnx" || base == onnxBase+".onnx_data" {
			result = append(result, f)
		}
	}

	return result
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("opening source: %w", err)
	}
	defer func() { _ = srcFile.Close() }()

	dstFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("creating destination: %w", err)
	}

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		_ = dstFile.Close()
		return fmt.Errorf("copying: %w", err)
	}

	return dstFile.Close()
}

// ValidVariants returns the list of valid ONNX variant names
func ValidVariants() []string {
	return []string{"", "fp16", "q4", "q4f16", "quantized"}
}

// IsValidVariant checks if a variant name is valid
func IsValidVariant(variant string) bool {
	return slices.Contains(ValidVariants(), variant)
}

// VariantDescription returns a human-readable description of a variant
func VariantDescription(variant string) string {
	switch variant {
	case "":
		return "full precision (default)"
	case "fp16":
		return "half precision (FP16)"
	case "q4":
		return "4-bit quantized"
	case "q4f16":
		return "4-bit quantized with FP16"
	case "quantized":
		return "INT8 quantized"
	default:
		return "unknown"
	}
}

// ListRepoFiles returns all files in a HuggingFace repo (useful for inspection)
func (c *HuggingFaceClient) ListRepoFiles(ctx context.Context, repoID string) ([]string, error) {
	repo := hub.New(repoID)
	if c.token != "" {
		repo = repo.WithAuth(c.token)
	}

	var files []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return nil, fmt.Errorf("listing files: %w", err)
		}
		files = append(files, fileName)
	}
	return files, nil
}

// DetectAvailableVariants returns which ONNX variants are available in a repo
func (c *HuggingFaceClient) DetectAvailableVariants(ctx context.Context, repoID string) ([]string, error) {
	files, err := c.ListRepoFiles(ctx, repoID)
	if err != nil {
		return nil, err
	}

	variants := []string{}
	variantPatterns := map[string]string{
		"":          "model.onnx",
		"fp16":      "model_fp16.onnx",
		"q4":        "model_q4.onnx",
		"q4f16":     "model_q4f16.onnx",
		"quantized": "model_quantized.onnx",
	}

	for variant, pattern := range variantPatterns {
		for _, f := range files {
			if filepath.Base(f) == pattern {
				if variant == "" {
					variants = append(variants, "default")
				} else {
					variants = append(variants, variant)
				}
				break
			}
		}
	}

	return variants, nil
}

// ParseHuggingFaceRef parses a model reference like "hf:owner/repo" and returns the repo ID
func ParseHuggingFaceRef(ref string) (repoID string, isHF bool) {
	if after, ok := strings.CutPrefix(ref, "hf:"); ok {
		return after, true
	}
	return "", false
}
