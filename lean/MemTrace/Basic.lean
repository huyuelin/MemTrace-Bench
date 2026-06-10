-- Basic types and definitions for MemTrace formal verification

-- Policy types as described in the paper Section 6
inductive Policy where
  | allow : Policy
  | scope : String → Policy
  | time : Nat → Policy
  | sensitivity : Nat → Policy
  | license : String → Policy
  | predicate : String → Policy  -- Simplified: use string identifier instead of function

-- Lattice levels for compiler output
inductive LatticeLevel where
  | fact : LatticeLevel
  | inference : LatticeLevel
  | synthesis : LatticeLevel

-- Memory segment with text, policy, and channel information
structure MemorySegment where
  text : String
  policy : Policy
  channel : String
  id : Nat

-- Certificate for validation
structure Certificate where
  policy : Policy
  memoryId : Nat
  hash : String
  isValid : Bool

-- Channel mediator type
structure Channel where
  name : String
  isMediated : Bool

-- Hash binding for integrity
structure HashBinding where
  memoryId : Nat
  hashValue : String
  isValid : Bool

-- Basic predicates (return Bool for decidability)
def allowed (p : Policy) (m : MemorySegment) : Bool :=
  match p with
  | .allow => true
  | .scope s => m.channel == s
  | .time t => false  -- simplified: would check timestamp
  | .sensitivity l => false  -- simplified: would check sensitivity level
  | .license l => false  -- simplified: would check license
  | .predicate f => false  -- simplified: would look up predicate by f

-- Fact in prompt check
def factInPrompt (cert : Certificate) (m : MemorySegment) : Prop :=
  cert.memoryId = m.id
