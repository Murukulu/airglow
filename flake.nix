{
  description = "Rust dev shell for airglow (aifsv2, gnn_leffingwell_odor)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer

            # grib -> proj -> proj-sys. Without a libproj that pkg-config can
            # find, proj-sys falls back to building PROJ from source (needs
            # cmake + sqlite, and takes minutes).
            pkg-config
            proj
          ];

          # rust-analyzer needs the stdlib sources to resolve std:: and offer
          # completions/goto-definition into core/alloc/std.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          # PROJ looks up proj.db here at runtime.
          PROJ_DATA = "${pkgs.proj}/share/proj";
        };
      });
    };
}
