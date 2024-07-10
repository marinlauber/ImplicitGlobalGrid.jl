using JLD2,Plots

file = jldopen("diffusion3D_cpu_KA.jld2", "r")
mygroup = file["case"]
@gif for i ∈ 1:50:10000
    plt = plot();
    T = Float32.(mygroup["T_$i"])
    contourf!(plt,axes(T,1),axes(T,2),T',color=:viridis)
end
# savefig("test.png")
