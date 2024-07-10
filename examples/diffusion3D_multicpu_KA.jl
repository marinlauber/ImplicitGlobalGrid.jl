using ImplicitGlobalGrid, JLD2
using KernelAbstractions: get_backend, @index, @kernel

@inline CI(a...) = CartesianIndex(a...)
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())
macro loop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    grab!(sym,ex)     # get arguments and replace composites in `ex`
    setdiff!(sym,[I]) # don't want to pass I as an argument
    @gensym kern      # generate unique kernel function name
    return quote
        @kernel function $kern($(rep.(sym)...),@Const(I0)) # replace composite arguments
            $I = @index(Global,Cartesian)
            $I += I0
            @fastmath @inbounds $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),$R[1]-oneunit($R[1]),ndrange=size($R))
    end |> esc
end
function grab!(sym,ex::Expr)
    ex.head == :. && return union!(sym,[ex])      # grab composite name and return
    start = ex.head==:(call) ? 2 : 1              # don't grab function names
    foreach(a->grab!(sym,a),ex.args[start:end])   # recurse into args
    ex.args[start:end] = rep.(ex.args[start:end]) # replace composites in args
end
grab!(sym,ex::Symbol) = union!(sym,[ex])        # grab symbol name
grab!(sym,ex) = nothing
rep(ex) = ex
rep(ex::Expr) = ex.head == :. ? Symbol(ex.args[2].value) : ex

# Physics
lam        = 1.0;                                       # Thermal conductivity
cp_min     = 1.0;                                       # Minimal heat capacity
lx, ly, lz = 10.0, 10.0, 10.0;                          # Length of computational domain in dimension x, y and z

# Numerics
nx, ny, nz = 128, 128, 128;                             # Number of gridpoints in dimensions x, y and z
nt         = 10000;                                     # Number of time steps
me, dims   = init_global_grid(nx, ny, nz);              # Initialize the implicit global grid
dx         = lx/(nx_g()-1);                             # Space step in dimension x
dy         = ly/(ny_g()-1);                             # ...        in dimension y
dz         = lz/(nz_g()-1);                             # ...        in dimension z

# Array initializations
T     = zeros(nx, ny, nz);
Cp    = zeros(nx, ny, nz);
dTedt = zeros(nx, ny, nz);
T_nohalo = zeros(nx-2, ny-2, nz-2); 

# Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
Cp .= cp_min .+ [5*exp(-((x_g(ix,dx,Cp)-lx/1.5))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) +
                 5*exp(-((x_g(ix,dx,Cp)-lx/3.0))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]
T  .= [100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
        50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]

# master communicator only has T_v
T_v = nothing
if me==0
    file = jldopen("diffusion3D_cpu_KA.jld2", "w")
    mygroup = JLD2.Group(file,"case")
    nx_v = (nx-2)*dims[1];
    ny_v = (ny-2)*dims[2];
    nz_v = (nz-2)*dims[3];
    T_v  = zeros(nx_v, ny_v, nz_v);
end

# Time loop
dt = min(dx*dx,dy*dy,dz*dz)*cp_min/lam/8.1;                                               # Time step for the 3D Heat diffusion
for it = 1:nt
    if mod(it, 50) == 1                                                                  # Visualize only every 500th time step
        T_nohalo .= T[2:end-1,2:end-1,2:end-1];                                           # Copy data removing the halo.
        gather!(T_nohalo, T_v)                                                            # Gather data on process 0 (could be interpolated/sampled first)
        if me==0 
            @show it
            mygroup["T_$it"] = T_v[:,:,nz_v÷2]
        end
    end
    dTedt .= 0.0;                                                                        # Reset the time derivative
    # compute diffusive flux
    for (i,di) ∈ zip([1,2,3],[dx,dy,dz])
        @loop dTedt[I] += lam/Cp[I]*(T[I+δ(i,I)] -2T[I] + T[I-δ(i,I)])/di/di over I ∈ CartesianIndices((2:nx-1,2:ny-1,2:nz-1))
    end
    # integrate in time
    @loop T[I] = (T[I] + dt*dTedt[I]) over I ∈ CartesianIndices((2:nx-1,2:ny-1,2:nz-1))
    update_halo!(T);                                                                      # Update the halo of T
end
# Finalize the implicit global grid
finalize_global_grid();  
# close file
me==0 && close(file)