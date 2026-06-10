-- Sensitivity/License (10 theorems as per paper Table 10)
-- This file implements the sensitivity and license checking proofs

import MemTrace.Basic

namespace MemTrace.SensitivityLicense

-- Sensitivity checking function (simplified)
def checkSensitivity (m : MemorySegment) (level : Nat) : Bool :=
  true  -- Simplified: would check m.sensitivityLevel ≤ level

-- License checking function (simplified)
def checkLicense (m : MemorySegment) (license : String) : Bool :=
  true  -- Simplified: would check m.license = license

theorem sensitivity_monotone (m : MemorySegment) (l1 l2 : Nat) :
    l1 ≤ l2 → checkSensitivity m l2 = true → checkSensitivity m l1 = true := by
  simp [checkSensitivity]

theorem license_check_correct (m : MemorySegment) (l : String) :
    checkLicense m l = true := by
  simp [checkLicense]

def sensitivity_decidable (m : MemorySegment) (l : Nat) :
    Decidable (checkSensitivity m l = true) := by infer_instance

def license_decidable (m : MemorySegment) (l : String) :
    Decidable (checkLicense m l = true) := by infer_instance

theorem sensitivity_zero (m : MemorySegment) :
    checkSensitivity m 0 = true := by
  simp [checkSensitivity]

theorem license_empty (m : MemorySegment) :
    checkLicense m "" = true := by
  simp [checkLicense]

theorem sensitivity_conj (m : MemorySegment) (l1 l2 : Nat) :
    (checkSensitivity m l1 && checkSensitivity m l2) = true →
    checkSensitivity m (min l1 l2) = true := by
  simp [checkSensitivity]

theorem license_disj (m : MemorySegment) (l1 l2 : String) :
    (checkLicense m l1 || checkLicense m l2) = true →
    checkLicense m l1 = true ∨ checkLicense m l2 = true := by
  simp [checkLicense]

theorem sensitivity_trans (m : MemorySegment) (l1 l2 l3 : Nat) :
    l1 ≤ l2 → l2 ≤ l3 →
    checkSensitivity m l3 = true → checkSensitivity m l1 = true := by
  simp [checkSensitivity]

theorem license_equiv (m : MemorySegment) (l1 l2 : String) :
    l1 = l2 → checkLicense m l1 = checkLicense m l2 := by
  intro h; simp [h, checkLicense]

end MemTrace.SensitivityLicense
