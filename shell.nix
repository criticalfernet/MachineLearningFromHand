{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
		python313Packages.matplotlib
		python313Packages.numpy
		python313Packages.pandas
  ];

  shellHook = ''
		fish
  '';
}
