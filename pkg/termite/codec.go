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
	"encoding/binary"
	"fmt"
	"io"
)

// SerializeFloatArrays converts a 2D float64 array to a byte slice.
func SerializeFloatArrays(w io.Writer, data [][]float32) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	for i, innerArray := range data {
		if i == 0 {
			if err := binary.Write(w, binary.LittleEndian, uint64(len(innerArray))); err != nil {
				return err
			}
		}
		for _, val := range innerArray {
			if err := binary.Write(w, binary.LittleEndian, val); err != nil {
				return err
			}
		}
	}
	return nil
}

// DeserializeFloatArrays reconstructs a 2D float64 array from a byte slice,
// given the dimensions of the original array.
func DeserializeFloatArrays(r io.Reader) ([][]float32, error) {
	var numVectors uint64
	if err := binary.Read(r, binary.LittleEndian, &numVectors); err != nil {
		return nil, fmt.Errorf("reading number of vectors: %w", err)
	}
	if numVectors == 0 {
		return [][]float32{}, nil
	}
	var dimension uint64
	if err := binary.Read(r, binary.LittleEndian, &dimension); err != nil {
		return nil, fmt.Errorf("reading number of vectors: %w", err)
	}
	result := make([][]float32, numVectors)
	for i := range numVectors {
		result[i] = make([]float32, dimension)
		for j := range dimension {
			if err := binary.Read(r, binary.LittleEndian, &result[i][j]); err != nil {
				return nil, fmt.Errorf("reading vector %d, dimension %d: %w", i, j, err)
			}
		}
	}
	return result, nil
}
