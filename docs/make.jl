using Documenter, PROPACK

makedocs(
  modules = [PROPACK],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    ansicolor = true,
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "PROPACK.jl",
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/PROPACK.jl.git",
  push_preview = true,
  devbranch = "main",
)
