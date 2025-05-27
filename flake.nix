{
  description = "Crazyflie firmware development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:brianmcgillion/nixpkgs/crazyflie-add";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        formatter = pkgs.alejandra;

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            gcc
            gnumake
            gcc-arm-embedded
            cfclient
            (python3.withPackages (
              p: with p; [
                cflib
              ]
            ))
          ];
        };
      }
    );
}
