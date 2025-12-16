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

package termite

import (
	"context"
	"encoding/binary"
	"sync/atomic"
	"time"

	"github.com/antflydb/antfly-go/libaf/reranking"
	"github.com/cespare/xxhash/v2"
	"github.com/jellydator/ttlcache/v3"
	"go.uber.org/zap"
	"golang.org/x/sync/singleflight"
)

// RerankingCacheTTL is the default TTL for cached reranking results
const RerankingCacheTTL = 2 * time.Minute

// CachedReranker wraps a reranker with caching support
type CachedReranker struct {
	reranker reranking.Model
	model    string
	cache    *ttlcache.Cache[string, []float32]
	sfGroup  *singleflight.Group
	logger   *zap.Logger

	// Metrics
	hits   atomic.Uint64
	misses atomic.Uint64
	sfHits atomic.Uint64
}

// NewCachedReranker wraps a reranker with caching
func NewCachedReranker(
	reranker reranking.Model,
	model string,
	cache *ttlcache.Cache[string, []float32],
	logger *zap.Logger,
) *CachedReranker {
	return &CachedReranker{
		reranker: reranker,
		model:    model,
		cache:    cache,
		sfGroup:  &singleflight.Group{},
		logger:   logger,
	}
}

// Rerank scores prompts with caching support
func (c *CachedReranker) Rerank(ctx context.Context, query string, prompts []string) ([]float32, error) {
	// Generate cache key from model + query + prompts hash
	key := c.cacheKey(query, prompts)

	// Check cache first
	if item := c.cache.Get(key); item != nil {
		c.hits.Add(1)
		RecordCacheHit("reranking")
		c.logger.Debug("Reranking cache hit",
			zap.String("model", c.model),
			zap.Int("num_prompts", len(prompts)))
		return item.Value(), nil
	}

	// Use singleflight to deduplicate concurrent identical requests
	result, err, shared := c.sfGroup.Do(key, func() (any, error) {
		c.misses.Add(1)
		RecordCacheMiss("reranking")

		start := time.Now()
		scores, err := c.reranker.Rerank(ctx, query, prompts)
		if err != nil {
			return nil, err
		}

		// Record duration
		RecordRequestDuration("rerank", c.model, "200", time.Since(start).Seconds())

		// Store in cache
		c.cache.Set(key, scores, ttlcache.DefaultTTL)

		c.logger.Debug("Reranking completed and cached",
			zap.String("model", c.model),
			zap.Int("num_prompts", len(prompts)),
			zap.Duration("duration", time.Since(start)))

		return scores, nil
	})

	if err != nil {
		return nil, err
	}

	if shared {
		c.sfHits.Add(1)
		c.logger.Debug("Singleflight hit for reranking request",
			zap.String("model", c.model))
	}

	return result.([]float32), nil
}

// cacheKey generates a unique cache key from model + query + prompts
func (c *CachedReranker) cacheKey(query string, prompts []string) string {
	h := xxhash.New()

	// Include model name
	_, _ = h.WriteString(c.model)
	_, _ = h.WriteString("|")

	// Include query
	_, _ = h.WriteString("q:")
	_, _ = h.WriteString(query)
	_, _ = h.WriteString("|")

	// Hash each prompt
	for i, prompt := range prompts {
		_, _ = h.WriteString("p")
		// Use index to ensure order matters
		_, _ = h.Write([]byte{byte(i >> 8), byte(i)})
		_, _ = h.WriteString(":")
		_, _ = h.WriteString(prompt)
		_, _ = h.WriteString("|")
	}

	// Convert uint64 hash to string key
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], h.Sum64())
	return string(buf[:])
}

// Close closes the underlying reranker
func (c *CachedReranker) Close() error {
	if closer, ok := c.reranker.(interface{ Close() error }); ok {
		return closer.Close()
	}
	return nil
}

// Stats returns cache statistics for this reranker
func (c *CachedReranker) Stats() RerankerCacheStats {
	return RerankerCacheStats{
		Model:            c.model,
		Hits:             c.hits.Load(),
		Misses:           c.misses.Load(),
		SingleflightHits: c.sfHits.Load(),
	}
}

// RerankerCacheStats holds cache statistics for a reranker
type RerankerCacheStats struct {
	Model            string `json:"model"`
	Hits             uint64 `json:"hits"`
	Misses           uint64 `json:"misses"`
	SingleflightHits uint64 `json:"singleflight_hits"`
}

// RerankingCache manages caching for multiple rerankers
type RerankingCache struct {
	cache  *ttlcache.Cache[string, []float32]
	logger *zap.Logger
	cancel context.CancelFunc
}

// NewRerankingCache creates a new reranking cache
func NewRerankingCache(logger *zap.Logger) *RerankingCache {
	cache := ttlcache.New(
		ttlcache.WithTTL[string, []float32](RerankingCacheTTL),
	)
	go cache.Start()

	ctx, cancel := context.WithCancel(context.Background())
	rc := &RerankingCache{
		cache:  cache,
		logger: logger,
		cancel: cancel,
	}

	// Log cache stats periodically
	go rc.logStats(ctx)

	return rc
}

// WrapReranker wraps a reranker with caching
func (rc *RerankingCache) WrapReranker(reranker reranking.Model, model string) *CachedReranker {
	return NewCachedReranker(reranker, model, rc.cache, rc.logger.Named(model))
}

// Close stops the cache
func (rc *RerankingCache) Close() {
	rc.cancel()
	rc.cache.Stop()
}

// logStats logs cache statistics periodically
func (rc *RerankingCache) logStats(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := rc.cache.Metrics()
			if metrics.Hits > 0 || metrics.Misses > 0 {
				hitRate := float64(0)
				total := metrics.Hits + metrics.Misses
				if total > 0 {
					hitRate = float64(metrics.Hits) / float64(total) * 100
				}
				rc.logger.Info("Reranking cache stats",
					zap.Uint64("hits", metrics.Hits),
					zap.Uint64("misses", metrics.Misses),
					zap.Float64("hit_rate_pct", hitRate),
					zap.Int("items", rc.cache.Len()))
			}
		}
	}
}

// Stats returns global cache statistics
func (rc *RerankingCache) Stats() map[string]any {
	metrics := rc.cache.Metrics()
	return map[string]any{
		"hits":   metrics.Hits,
		"misses": metrics.Misses,
		"items":  rc.cache.Len(),
	}
}
