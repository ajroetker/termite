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

package v1alpha1

import (
	"fmt"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
)

// ValidateCreate validates the pool configuration when creating a new pool
func (r *TermitePool) ValidateCreate() error {
	return r.validateTermitePool()
}

// ValidateUpdate validates the pool configuration when updating an existing pool
func (r *TermitePool) ValidateUpdate(old runtime.Object) error {
	oldPool := old.(*TermitePool)
	if err := r.validateImmutability(oldPool); err != nil {
		return err
	}
	return r.validateTermitePool()
}

// ValidateDelete validates pool deletion (no validation needed)
func (r *TermitePool) ValidateDelete() error {
	// No validation needed for delete operations
	return nil
}

// validateTermitePool performs all validation checks
func (r *TermitePool) validateTermitePool() error {
	var allErrors []string

	if err := r.validateGKEConfig(); err != nil {
		allErrors = append(allErrors, err.Error())
	}

	if err := r.validateNoConflictingSettings(); err != nil {
		allErrors = append(allErrors, err.Error())
	}

	if err := r.validateReplicaCounts(); err != nil {
		allErrors = append(allErrors, err.Error())
	}

	if len(allErrors) > 0 {
		return fmt.Errorf("TermitePool validation failed:\n  - %s",
			strings.Join(allErrors, "\n  - "))
	}

	return nil
}

// validateGKEConfig validates GKE-specific configuration
func (r *TermitePool) validateGKEConfig() error {
	if r.Spec.GKE == nil {
		return nil
	}

	gke := r.Spec.GKE

	// Validate compute class enum (only if non-empty)
	if gke.AutopilotComputeClass != "" {
		validClasses := []string{"Accelerator", "Balanced", "Performance", "Scale-Out", "autopilot", "autopilot-spot"}
		valid := slices.Contains(validClasses, gke.AutopilotComputeClass)
		if !valid {
			return fmt.Errorf("invalid GKE Autopilot compute class '%s'. Must be one of: %s",
				gke.AutopilotComputeClass, strings.Join(validClasses, ", "))
		}
	}

	// Validate compute class requires Autopilot
	if gke.AutopilotComputeClass != "" && !gke.Autopilot {
		return fmt.Errorf(`spec.gke.autopilotComputeClass is set but spec.gke.autopilot=false

Problem: Compute classes only work with GKE Autopilot clusters.

Solution: Either:
  Option 1 (Use Autopilot): Set spec.gke.autopilot=true
  Option 2 (Standard GKE): Remove spec.gke.autopilotComputeClass and use spec.hardware.spot instead`)
	}

	// Validate Accelerator compute class requires GPU (NOT TPU)
	// TPU workloads should NOT use Accelerator class - they use node selectors instead
	if gke.AutopilotComputeClass == "Accelerator" {
		hasGPU := r.hasGPUResources()

		if !hasGPU {
			return fmt.Errorf(`spec.gke.autopilotComputeClass='Accelerator' requires GPU resources

Problem: GKE Autopilot's Accelerator compute class is for GPU workloads ONLY.
For TPU workloads, do NOT use 'Accelerator' class - TPU provisioning uses node selectors.

Solution for GPU workloads: Add GPU resources to spec.resources
Solution for TPU workloads: Remove autopilotComputeClass='Accelerator' and use TPU node selectors

Example (GPU):
  spec:
    resources:
      limits:
        nvidia.com/gpu: "1"
    gke:
      autopilot: true
      autopilotComputeClass: "Accelerator"

Example (TPU with Spot pricing):
  spec:
    hardware:
      accelerator: "tpu-v4-podslice"
      topology: "2x2x1"
    gke:
      autopilot: true
      autopilotComputeClass: "autopilot-spot"  # Use this for spot, NOT "Accelerator"
    resources:
      limits:
        google.com/tpu: "4"`)
		}
	}

	return nil
}

// validateNoConflictingSettings validates that hardware.spot doesn't conflict with Autopilot
func (r *TermitePool) validateNoConflictingSettings() error {
	if r.Spec.GKE == nil || !r.Spec.GKE.Autopilot {
		return nil
	}

	// Check hardware.spot conflicts with Autopilot
	if r.Spec.Hardware.Spot {
		return fmt.Errorf(`spec.hardware.spot=true conflicts with spec.gke.autopilot=true

Problem: GKE Autopilot uses compute classes for spot scheduling, not node selectors.

Solution: Remove 'hardware.spot: true' and use 'gke.autopilotComputeClass: autopilot-spot' instead

Example:
  spec:
    hardware:
      # spot: true  # REMOVE THIS
      accelerator: "tpu-v5-lite-podslice"
      topology: "2x2"
    gke:
      autopilot: true
      autopilotComputeClass: 'autopilot-spot'  # ADD THIS`)
	}

	return nil
}

// validateReplicaCounts validates that replica counts are valid
func (r *TermitePool) validateReplicaCounts() error {
	if r.Spec.Replicas.Min < 0 {
		return fmt.Errorf("spec.replicas.min must be >= 0, got %d", r.Spec.Replicas.Min)
	}

	if r.Spec.Replicas.Max <= 0 {
		return fmt.Errorf("spec.replicas.max must be > 0, got %d", r.Spec.Replicas.Max)
	}

	if r.Spec.Replicas.Min > r.Spec.Replicas.Max {
		return fmt.Errorf("spec.replicas.min (%d) cannot be greater than spec.replicas.max (%d)",
			r.Spec.Replicas.Min, r.Spec.Replicas.Max)
	}

	return nil
}

// validateImmutability validates that immutable fields haven't changed
func (r *TermitePool) validateImmutability(old *TermitePool) error {
	var errors []string

	// Check if both old and new have GKE config
	if r.Spec.GKE != nil && old.Spec.GKE != nil {
		// Check Autopilot mode immutability
		if r.Spec.GKE.Autopilot != old.Spec.GKE.Autopilot {
			errors = append(errors, fmt.Sprintf(
				`field 'spec.gke.autopilot' is immutable after deployment

Problem: Changing Autopilot mode requires pod recreation, which may disrupt model serving.

Solution: Delete and recreate the pool to change this setting.

Current value: %v
Attempted change: %v`,
				old.Spec.GKE.Autopilot, r.Spec.GKE.Autopilot))
		}

		// Check compute class immutability (only when Autopilot is enabled)
		if r.Spec.GKE.Autopilot && r.Spec.GKE.AutopilotComputeClass != old.Spec.GKE.AutopilotComputeClass {
			errors = append(errors, fmt.Sprintf(
				`field 'spec.gke.autopilotComputeClass' is immutable after deployment

Problem: Changing compute class requires pod recreation, which may disrupt model serving.

Solution: Delete and recreate the pool to change this setting.

Current value: "%s"
Attempted change: "%s"`,
				old.Spec.GKE.AutopilotComputeClass, r.Spec.GKE.AutopilotComputeClass))
		}
	}

	// Handle case where GKE config is being added/removed after creation
	if (r.Spec.GKE != nil && old.Spec.GKE == nil) || (r.Spec.GKE == nil && old.Spec.GKE != nil) {
		if old.Spec.GKE != nil && old.Spec.GKE.Autopilot {
			errors = append(errors, `cannot remove spec.gke configuration after deployment when autopilot was enabled

Problem: Removing GKE configuration would change the scheduling behavior.

Solution: Delete and recreate the pool to change this setting.`)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("%s", strings.Join(errors, "\n\n"))
	}

	return nil
}

// hasGPUResources checks if GPU resources are present in spec.resources
func (r *TermitePool) hasGPUResources() bool {
	if r.Spec.Resources == nil || r.Spec.Resources.Limits == nil {
		return false
	}
	_, hasNvidiaGPU := r.Spec.Resources.Limits["nvidia.com/gpu"]
	_, hasGoogleGPU := r.Spec.Resources.Limits["cloud.google.com/gke-gpu"]
	return hasNvidiaGPU || hasGoogleGPU
}
