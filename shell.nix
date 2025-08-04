{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
		python313Packages.matplotlib
		python313Packages.numpy
		python313Packages.pandas
		python313Packages.scipy
  ];

  shellHook = ''
		fish
  '';
}
