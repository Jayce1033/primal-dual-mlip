"""Construct the configured ACE basis and optional reference energies."""
function build_model(config::AbstractDict)
    spec = config["model"]
    kwargs = (elements=Symbol.(spec["elements"]), order=Int(spec["order"]),
              totaldegree=Int(spec["totaldegree"]), rcut=Float64(spec["rcut"]))
    if haskey(spec, "Eref")
        reference = [Symbol(element) => Float64(energy) for (element, energy) in sort!(collect(spec["Eref"]); by=first)]
        return ACEpotentials.ace1_model(; kwargs..., Eref=reference)
    end
    return ACEpotentials.ace1_model(; kwargs...)
end
