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

//go:build onnx && ORT

package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/antflydb/antfly-go/libaf/ai"
	libafembed "github.com/antflydb/antfly-go/libaf/embeddings"
	ort "github.com/yalue/onnxruntime_go"
	"go.uber.org/zap"
	_ "golang.org/x/image/webp"
)

// CLIPEmbedder implements multimodal embeddings using CLIP ONNX models.
// It can embed both images and text into a shared embedding space where
// image-text similarity can be computed via cosine similarity.
//
// Build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type CLIPEmbedder struct {
	visualModelPath      string
	textModelPath        string
	visualProjectionPath string
	textProjectionPath   string
	tokenizer            *CLIPTokenizer
	config               *CLIPConfig
	logger               *zap.Logger
	caps                 libafembed.EmbedderCapabilities
	modelPath            string
	mu                   sync.Mutex // Protects session operations
}

// CLIPConfig holds the CLIP model configuration
type CLIPConfig struct {
	ModelType     string           `json:"model_type"`
	VisionConfig  CLIPVisionConfig `json:"vision_config"`
	TextConfig    CLIPTextConfig   `json:"text_config"`
	ProjectionDim int              `json:"projection_dim"`
}

// CLIPVisionConfig holds vision encoder configuration
type CLIPVisionConfig struct {
	HiddenSize    int `json:"hidden_size"`
	ImageSize     int `json:"image_size"`
	PatchSize     int `json:"patch_size"`
	ProjectionDim int `json:"projection_dim"`
}

// CLIPTextConfig holds text encoder configuration
type CLIPTextConfig struct {
	HiddenSize            int `json:"hidden_size"`
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
	ProjectionDim         int `json:"projection_dim"`
}

// CLIPTokenizer is a simple tokenizer for CLIP text encoding
type CLIPTokenizer struct {
	Vocab       map[string]int `json:"vocab"`
	MergesRules []string       `json:"merges"`
	MaxLength   int
	PadTokenID  int
	EOSTokenID  int
	BOSTokenID  int
}

// PreprocessorConfig holds image preprocessing configuration
type PreprocessorConfig struct {
	DoResize      bool      `json:"do_resize"`
	Size          ImageSize `json:"size"`
	DoRescale     bool      `json:"do_rescale"`
	RescaleFactor float32   `json:"rescale_factor"`
	DoNormalize   bool      `json:"do_normalize"`
	ImageMean     []float32 `json:"image_mean"`
	ImageStd      []float32 `json:"image_std"`
	DoCenterCrop  bool      `json:"do_center_crop"`
	CropSize      ImageSize `json:"crop_size"`
	DoConvertRGB  bool      `json:"do_convert_rgb"`
}

// ImageSize can be either an int or a struct with width/height
type ImageSize struct {
	ShortestEdge int `json:"shortest_edge,omitempty"`
	Height       int `json:"height,omitempty"`
	Width        int `json:"width,omitempty"`
}

// ONNX Runtime initialization
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func initONNXRuntime() error {
	ortInitOnce.Do(func() {
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// NewCLIPEmbedder creates a new CLIP embedder from a model directory.
// The directory should contain:
//   - visual_model.onnx (or visual_model_quantized.onnx)
//   - text_model.onnx (or text_model_quantized.onnx)
//   - clip_config.json or config.json
//   - preprocessor_config.json
//   - tokenizer.json
//
// Build with -tags="onnx,ORT" to enable this embedder.
func NewCLIPEmbedder(modelPath string, quantized bool, logger *zap.Logger) (*CLIPEmbedder, error) {
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}

	if logger == nil {
		logger = zap.NewNop()
	}

	logger.Info("Initializing CLIP embedder",
		zap.String("modelPath", modelPath),
		zap.Bool("quantized", quantized))

	// Load configuration
	config, err := loadCLIPConfig(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading CLIP config: %w", err)
	}

	// Determine ONNX filenames
	visualFile := "visual_model.onnx"
	textFile := "text_model.onnx"
	if quantized {
		visualFile = "visual_model_quantized.onnx"
		textFile = "text_model_quantized.onnx"
	}

	visualPath := filepath.Join(modelPath, visualFile)
	textPath := filepath.Join(modelPath, textFile)
	visualProjectionPath := filepath.Join(modelPath, "visual_projection.onnx")
	textProjectionPath := filepath.Join(modelPath, "text_projection.onnx")

	// Verify files exist
	if _, err := os.Stat(visualPath); err != nil {
		return nil, fmt.Errorf("visual model not found: %s", visualPath)
	}
	if _, err := os.Stat(textPath); err != nil {
		return nil, fmt.Errorf("text model not found: %s", textPath)
	}
	// Check for projection layers (required for proper embedding projection)
	hasProjections := true
	if _, err := os.Stat(visualProjectionPath); err != nil {
		hasProjections = false
		logger.Warn("visual projection not found, embeddings may have mismatched dimensions",
			zap.String("path", visualProjectionPath))
	}
	if _, err := os.Stat(textProjectionPath); err != nil {
		hasProjections = false
		logger.Warn("text projection not found, embeddings may have mismatched dimensions",
			zap.String("path", textProjectionPath))
	}
	if !hasProjections {
		visualProjectionPath = ""
		textProjectionPath = ""
	}

	// Initialize ONNX Runtime
	if err := initONNXRuntime(); err != nil {
		return nil, fmt.Errorf("initializing ONNX runtime: %w", err)
	}

	// Load tokenizer
	tokenizer, err := loadCLIPTokenizer(modelPath)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	// Determine image size from config
	imageSize := 224
	if config.VisionConfig.ImageSize > 0 {
		imageSize = config.VisionConfig.ImageSize
	}

	logger.Info("CLIP embedder initialized",
		zap.Int("projectionDim", config.ProjectionDim),
		zap.Int("imageSize", imageSize))

	return &CLIPEmbedder{
		visualModelPath:      visualPath,
		textModelPath:        textPath,
		visualProjectionPath: visualProjectionPath,
		textProjectionPath:   textProjectionPath,
		tokenizer:            tokenizer,
		config:               config,
		logger:               logger,
		modelPath:            modelPath,
		caps: libafembed.EmbedderCapabilities{
			SupportedMIMETypes: []libafembed.MIMETypeSupport{
				{MIMEType: "text/plain"},
				{MIMEType: "image/png"},
				{MIMEType: "image/jpeg"},
				{MIMEType: "image/gif"},
				{MIMEType: "image/webp"},
			},
			Dimensions:       []int{config.ProjectionDim},
			DefaultDimension: config.ProjectionDim,
			SupportsFusion:   false, // CLIP creates separate embeddings, not fused
		},
	}, nil
}

// Capabilities returns the embedder capabilities
func (c *CLIPEmbedder) Capabilities() libafembed.EmbedderCapabilities {
	return c.caps
}

// Embed generates embeddings for the given content.
// For text content, uses the text encoder.
// For image content (BinaryContent), uses the visual encoder.
func (c *CLIPEmbedder) Embed(ctx context.Context, contents [][]ai.ContentPart) ([][]float32, error) {
	if len(contents) == 0 {
		return [][]float32{}, nil
	}

	embeddings := make([][]float32, len(contents))

	for i, parts := range contents {
		var embedding []float32
		var err error

		for _, part := range parts {
			switch p := part.(type) {
			case ai.BinaryContent:
				if strings.HasPrefix(p.MIMEType, "image/") {
					embedding, err = c.embedImage(p.Data)
					if err != nil {
						return nil, fmt.Errorf("embedding image at index %d: %w", i, err)
					}
				}
			case ai.TextContent:
				embedding, err = c.embedText(p.Text)
				if err != nil {
					return nil, fmt.Errorf("embedding text at index %d: %w", i, err)
				}
			}

			if embedding != nil {
				break
			}
		}

		if embedding == nil {
			return nil, fmt.Errorf("no valid content found at index %d", i)
		}

		embeddings[i] = embedding
	}

	return embeddings, nil
}

// embedImage processes an image and returns its embedding
func (c *CLIPEmbedder) embedImage(imageData []byte) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Decode image
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("decoding image: %w", err)
	}

	// Get target size from config
	targetSize := 224
	if c.config.VisionConfig.ImageSize > 0 {
		targetSize = c.config.VisionConfig.ImageSize
	}

	// Preprocess image to tensor
	pixelValues := preprocessImage(img, targetSize)

	// Create input tensor [1, 3, H, W]
	inputShape := ort.NewShape(1, 3, int64(targetSize), int64(targetSize))
	inputTensor, err := ort.NewTensor(inputShape, pixelValues)
	if err != nil {
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensors
	// Visual model outputs: last_hidden_state [1, num_patches, hidden_size] and pooler_output [1, hidden_size]
	hiddenSize := int64(c.config.VisionConfig.HiddenSize)
	if hiddenSize == 0 {
		hiddenSize = 768 // Default for ViT-B
	}

	// We only need pooler_output for embeddings
	outputShape := ort.NewShape(1, hiddenSize)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create and run session
	session, err := ort.NewAdvancedSession(
		c.visualModelPath,
		[]string{"pixel_values"},
		[]string{"pooler_output"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("creating visual session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("running visual inference: %w", err)
	}

	// Get output data
	outputData := outputTensor.GetData()
	embedding := make([]float32, len(outputData))
	copy(embedding, outputData)

	// Apply visual projection if available
	if c.visualProjectionPath != "" {
		projected, err := c.applyProjection(c.visualProjectionPath, embedding, hiddenSize, int64(c.config.ProjectionDim))
		if err != nil {
			return nil, fmt.Errorf("applying visual projection: %w", err)
		}
		embedding = projected
	}

	// Normalize embedding
	return normalizeL2(embedding), nil
}

// embedText tokenizes text and returns its embedding
func (c *CLIPEmbedder) embedText(text string) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Tokenize text
	inputIDs, attentionMask := c.tokenizer.Encode(text)
	seqLen := int64(len(inputIDs))

	// Convert to int64 for ONNX
	inputIDs64 := make([]int64, len(inputIDs))
	attMask64 := make([]int64, len(attentionMask))
	for i := range inputIDs {
		inputIDs64[i] = int64(inputIDs[i])
		attMask64[i] = int64(attentionMask[i])
	}

	// Create input tensors [1, seq_len]
	inputIDsShape := ort.NewShape(1, seqLen)
	inputIDsTensor, err := ort.NewTensor(inputIDsShape, inputIDs64)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attMaskTensor, err := ort.NewTensor(inputIDsShape, attMask64)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer attMaskTensor.Destroy()

	// Create output tensor
	hiddenSize := int64(c.config.TextConfig.HiddenSize)
	if hiddenSize == 0 {
		hiddenSize = 512 // Default for CLIP text encoder
	}

	outputShape := ort.NewShape(1, hiddenSize)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create and run session
	session, err := ort.NewAdvancedSession(
		c.textModelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"pooler_output"},
		[]ort.Value{inputIDsTensor, attMaskTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("creating text session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("running text inference: %w", err)
	}

	// Get output data
	outputData := outputTensor.GetData()
	embedding := make([]float32, len(outputData))
	copy(embedding, outputData)

	// Apply text projection if available
	if c.textProjectionPath != "" {
		projected, err := c.applyProjection(c.textProjectionPath, embedding, hiddenSize, int64(c.config.ProjectionDim))
		if err != nil {
			return nil, fmt.Errorf("applying text projection: %w", err)
		}
		embedding = projected
	}

	// Normalize embedding
	return normalizeL2(embedding), nil
}

// applyProjection runs an embedding through a projection ONNX model
func (c *CLIPEmbedder) applyProjection(projPath string, input []float32, inputDim, outputDim int64) ([]float32, error) {
	// Create input tensor [1, inputDim]
	inputShape := ort.NewShape(1, inputDim)
	inputTensor, err := ort.NewTensor(inputShape, input)
	if err != nil {
		return nil, fmt.Errorf("creating projection input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor [1, outputDim]
	outputShape := ort.NewShape(1, outputDim)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating projection output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Create and run projection session
	session, err := ort.NewAdvancedSession(
		projPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("creating projection session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("running projection: %w", err)
	}

	// Copy output
	outputData := outputTensor.GetData()
	projected := make([]float32, len(outputData))
	copy(projected, outputData)

	return projected, nil
}

// Close releases resources
func (c *CLIPEmbedder) Close() error {
	// No persistent sessions to close in this implementation
	return nil
}

// Helper functions

func loadCLIPConfig(modelPath string) (*CLIPConfig, error) {
	configPaths := []string{
		filepath.Join(modelPath, "clip_config.json"),
		filepath.Join(modelPath, "config.json"),
	}

	var config CLIPConfig
	for _, path := range configPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		if err := json.Unmarshal(data, &config); err != nil {
			continue
		}

		// Check if it's a valid CLIP config
		if config.ProjectionDim > 0 || config.VisionConfig.ProjectionDim > 0 {
			if config.ProjectionDim == 0 {
				config.ProjectionDim = config.VisionConfig.ProjectionDim
			}
			return &config, nil
		}
	}

	// Return default config for CLIP ViT-B/32
	return &CLIPConfig{
		ModelType:     "clip",
		ProjectionDim: 512,
		VisionConfig: CLIPVisionConfig{
			HiddenSize:    768,
			ImageSize:     224,
			PatchSize:     32,
			ProjectionDim: 512,
		},
		TextConfig: CLIPTextConfig{
			HiddenSize:            512,
			MaxPositionEmbeddings: 77,
			ProjectionDim:         512,
		},
	}, nil
}

func loadCLIPTokenizer(modelPath string) (*CLIPTokenizer, error) {
	tokenizerPath := filepath.Join(modelPath, "tokenizer.json")
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("reading tokenizer.json: %w", err)
	}

	var tokenizerData struct {
		Model struct {
			Vocab  map[string]int `json:"vocab"`
			Merges []string       `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &tokenizerData); err != nil {
		return nil, fmt.Errorf("parsing tokenizer.json: %w", err)
	}

	tokenizer := &CLIPTokenizer{
		Vocab:       tokenizerData.Model.Vocab,
		MergesRules: tokenizerData.Model.Merges,
		MaxLength:   77, // CLIP's default max length
		PadTokenID:  0,
		EOSTokenID:  49407, // <|endoftext|>
		BOSTokenID:  49406, // <|startoftext|>
	}

	// Find special token IDs from added_tokens
	for _, token := range tokenizerData.AddedTokens {
		switch token.Content {
		case "<|endoftext|>":
			tokenizer.EOSTokenID = token.ID
		case "<|startoftext|>":
			tokenizer.BOSTokenID = token.ID
		}
	}

	return tokenizer, nil
}

// Encode tokenizes text for CLIP.
// Returns input_ids and attention_mask.
// Note: This is a simplified tokenizer. For production, use a proper BPE implementation.
func (t *CLIPTokenizer) Encode(text string) ([]int, []int) {
	text = strings.ToLower(text)
	words := strings.Fields(text)

	// Start with BOS token
	inputIDs := []int{t.BOSTokenID}

	// Tokenize each word
	for _, word := range words {
		// Add space prefix for BPE compatibility
		wordWithSpace := " " + word
		if id, ok := t.Vocab[wordWithSpace]; ok {
			inputIDs = append(inputIDs, id)
		} else {
			// Try without space prefix
			if id, ok := t.Vocab[word]; ok {
				inputIDs = append(inputIDs, id)
			} else {
				// Character-level fallback
				for _, char := range word {
					if id, ok := t.Vocab[string(char)]; ok {
						inputIDs = append(inputIDs, id)
					}
				}
			}
		}
	}

	// Add EOS token
	inputIDs = append(inputIDs, t.EOSTokenID)

	// Truncate if needed
	if len(inputIDs) > t.MaxLength {
		inputIDs = inputIDs[:t.MaxLength-1]
		inputIDs = append(inputIDs, t.EOSTokenID)
	}

	// Create attention mask and pad
	attentionMask := make([]int, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	for len(inputIDs) < t.MaxLength {
		inputIDs = append(inputIDs, t.PadTokenID)
		attentionMask = append(attentionMask, 0)
	}

	return inputIDs, attentionMask
}

// preprocessImage resizes and normalizes an image for CLIP
func preprocessImage(img image.Image, targetSize int) []float32 {
	// CLIP normalization values
	mean := []float32{0.48145466, 0.4578275, 0.40821073}
	std := []float32{0.26862954, 0.26130258, 0.27577711}

	// Resize image
	resized := resizeImage(img, targetSize, targetSize)

	// Convert to float32 tensor in [C, H, W] format
	pixels := make([]float32, 3*targetSize*targetSize)

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			// Convert to 0-1 range and normalize
			rf := float32(r>>8) / 255.0
			gf := float32(g>>8) / 255.0
			bf := float32(b>>8) / 255.0

			// Apply normalization
			rf = (rf - mean[0]) / std[0]
			gf = (gf - mean[1]) / std[1]
			bf = (bf - mean[2]) / std[2]

			// Store in CHW format
			idx := y*targetSize + x
			pixels[0*targetSize*targetSize+idx] = rf // R channel
			pixels[1*targetSize*targetSize+idx] = gf // G channel
			pixels[2*targetSize*targetSize+idx] = bf // B channel
		}
	}

	return pixels
}

// resizeImage performs nearest-neighbor resize
func resizeImage(img image.Image, width, height int) image.Image {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, width, height))

	xRatio := float64(srcW) / float64(width)
	yRatio := float64(srcH) / float64(height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			srcX := int(float64(x) * xRatio)
			srcY := int(float64(y) * yRatio)

			if srcX >= srcW {
				srcX = srcW - 1
			}
			if srcY >= srcH {
				srcY = srcH - 1
			}

			dst.Set(x, y, img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY))
		}
	}

	return dst
}
