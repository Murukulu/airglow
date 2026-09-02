{
  description = "Rust dev shell + package for airglow (aifsv2, gnn_leffingwell_odor)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: {
        default = self.packages.${pkgs.system}.aifsv2;

        aifsv2 = pkgs.rustPlatform.buildRustPackage {
          pname = "aifsv2";
          version = "0.1.0";

          src = ./aifsv2;

          cargoLock = {
            lockFile = ./aifsv2/Cargo.lock;
          };

          # pkg-config finds libproj/libeccodes; bindgenHook wires up libclang
          # (LIBCLANG_PATH + include flags) for eccodes-sys's bindgen step.
          nativeBuildInputs = with pkgs; [
            pkg-config
            rustPlatform.bindgenHook
            makeWrapper
          ];

          buildInputs = with pkgs; [
            proj
            eccodes
          ];

          # The GPU/eccodes tests need hardware and data files not present in the
          # build sandbox; skip them for the package build.
          doCheck = false;

          # PROJ looks up proj.db here at runtime; the wgpu backend dlopens the
          # Vulkan loader. Baking both into the wrapped binary lets `nix run`
          # work without entering the dev shell.
          postInstall = ''
            wrapProgram $out/bin/aifsv2 \
              --set-default PROJ_DATA "${pkgs.proj}/share/proj" \
              --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ]}"
          '';
        };
      });

      apps = forAllSystems (pkgs: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.aifsv2}/bin/aifsv2";
        };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer
            # lspmux proxy: one persistent rust-analyzer per project, shared
            # across editor windows/restarts (binary is ra-multiplex).
            ra-multiplex

            # grib -> proj -> proj-sys.
            # Without a libproj that pkg-config can find, proj-sys falls
            # back to building PROJ from source.
            pkg-config
            proj

            # eccodes -> eccodes-sys. That crate deliberately refuses to build
            # ecCodes itself: it pkg-config probes for an installed libeccodes
            # (>= 2.24.0) and fails the build if there is none. nixpkgs patches
            # eccodes.pc to carry absolute store paths, so no PKG_CONFIG_PATH.
            eccodes

            # eccodes-sys generates its bindings with bindgen, which dlopens
            # libclang at build time.
            llvmPackages.libclang
          ];

          # rust-analyzer needs the stdlib sources to resolve std:: and offer
          # completions/goto-definition into core/alloc/std.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          # PROJ looks up proj.db here at runtime.
          PROJ_DATA = "${pkgs.proj}/share/proj";

          # Where bindgen looks for libclang.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # cudarc dlopens libcuda (driver) and libnvrtc (toolkit) at runtime. The nix
          # glibc does not read /etc/ld.so.cache, and putting all of /usr/lib on
          # LD_LIBRARY_PATH would shadow nix libraries, so expose exactly the CUDA libs
          # through a symlink shim. Detection has to happen here at shell entry, not
          # at eval time: the flake is pure and the same shell must work on hosts
          # without CUDA, which skip the shim entirely.
          shellHook = ''
            if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
              cuda_shim=/var/tmp/$USER/cuda-shim
              mkdir -p "$cuda_shim"
              # libcuda plus every libnvidia-* companion: the driver dlopens its own
              # helpers by name at JIT time (ptxjitcompiler, nvvm, gpucomp, ...), and
              # none of them shadow anything nix provides.
              for lib in /usr/lib/x86_64-linux-gnu/libcuda.so* \
                         /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
                         /usr/local/cuda/lib64/libnvrtc*.so*; do
                [ -e "$lib" ] && ln -sf "$lib" "$cuda_shim/"
              done
              export LD_LIBRARY_PATH=$cuda_shim''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            fi
          '';
        };
      });
    };
}
