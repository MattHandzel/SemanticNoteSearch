{pkgs ? import <nixpkgs> {}}:
with pkgs;
  mkShell {
    buildInputs = [
      python311Packages.virtualenv
      python311Packages.sentence-transformers
      python311Packages.qdrant-client
      python311Packages.python-dotenv
      python311Packages.flask

      python311

      docker
      docker-compose
    ];

    shellHook = ''
      if ! [ -e .venv ]; then
        python3 -m venv .venv
        pip install python-frontmatter
      fi
      source .venv/bin/activate


    '';
  }
