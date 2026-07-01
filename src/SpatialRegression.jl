module SpatialRegression

using LinearAlgebra
using Statistics
using StaticArrays
using ForwardDiff
using DelaunayTriangulation
using OhMyThreads: @localize, @tasks, @set, tmap, tforeach, ChannelLike
using ChunkSplitters
using GLM: glm, coef, dispersion

import Distributions
import Accessors

# Imports for writing custom chunkable iterators
import Base:
  iterate, length, eltype,
  firstindex, lastindex,
  view

import ChunkSplitters: is_chunkable

# Isolate Distributions + GLM code in separate module.
# Any required imports are handled in the module and available here.
include("GLMUtilities.jl")
include("utilities.jl")
using .GLMUtilities

# Export adaptive refinement tools
include("AdaptiveRefinement.jl")
import .AdaptiveRefinement
export AdaptiveRefinement

abstract type AbstractVertexModel end
abstract type AbstractPenalty end
abstract type AbstractSpatialModel end
include("TriangleObs.jl")
include("SpatialVertexModel.jl")
include("VertexGLM.jl")
include("VertexQREG.jl")
export TriangleObs,
  VertexGLM,
  SpatialVertexModel

include("penalty.jl")
export L2Squared, L1Approx

end
