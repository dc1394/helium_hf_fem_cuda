module Helium_HF_FEM_SCF
    include("helium_hf_fem_eigen.jl")
    include("helium_vh_fem.jl")
    using Printf
    using .Helium_HF_FEM_Eigen
    using .Helium_Vh_FEM

    const DR = 0.01
    const MAXITER = 1000
    const DOUBLE_THRESHOLD = 1.0E-4
    const THRESHOLD = 1.0E-9

    function scfloop()
        # Schrödinger方程式を解くルーチンの初期処理
        hfem_param, hfem_val = Helium_HF_FEM_Eigen.construct()

        # 倍精度浮動小数点で計算する際のフラグ
        double_flag = false
        
        # 有限要素法のデータのみ生成
        Helium_HF_FEM_Eigen.make_wavefunction(double_flag, 0, hfem_param, hfem_val, nothing)
        
        # Poisson方程式を解くルーチンの初期処理
        vh_param, vh_val = Helium_Vh_FEM.construct(hfem_param)
        
        # 仮の電子密度でPoisson方程式を解く
        Helium_Vh_FEM.solvepoisson!(0, hfem_param, hfem_val, vh_val)

        # 新しく計算されたエネルギー
        enew = 0.0

        for iter in 1:MAXITER
            eigenenergy = Helium_HF_FEM_Eigen.make_wavefunction(double_flag, iter, hfem_param, hfem_val, vh_val)
            
            # 前回のSCF計算のエネルギーを保管
            eold = enew
            
            # 今回のSCF計算のエネルギーを計算する
            enew = Helium_HF_FEM_Eigen.get_totalenergy(eigenenergy, hfem_param, hfem_val, vh_val)
            
            if double_flag
                @printf "(double): Iteration # %2d: HF eigenvalue = %.14f, energy = %.14f\n" iter eigenenergy enew
            else
                @printf "(float): Iteration # %2d: HF eigenvalue = %.14f, energy = %.14f\n" iter eigenenergy enew
            end
            
            # 今回のSCF計算のエネルギーと前回のSCF計算のエネルギーの差の絶対値
            ediff = abs(enew - eold)

            # SCF計算が収束したかどうか
            if ediff <= THRESHOLD
                # 波動関数を規格化
                hfem_val.phi ./= sqrt(4.0 * pi)

                # 収束したのでhfem_param, hfem_val, エネルギーを返す
                return hfem_param, hfem_val, enew
            elseif ediff <= DOUBLE_THRESHOLD
                double_flag = true
            end
            
            # Poisson方程式を解く
            Helium_Vh_FEM.solvepoisson!(scfloop, hfem_param, hfem_val, vh_val)
        end

        return nothing
    end

    save_result(hfem_param, hfem_val, filename) = let
        open(filename, "w" ) do fp
            len = length(hfem_val.node_r_glo)
            imax = floor(Int, (hfem_val.node_r_glo[len] - hfem_val.node_r_glo[1]) / DR)
            for i = 0:imax
                r = hfem_val.node_r_glo[1] + DR * float(i)
                println(fp, @sprintf "%.14f, %.14f" r Helium_Vh_FEM.phi(hfem_param, hfem_val, r))
            end
        end
    end
end

