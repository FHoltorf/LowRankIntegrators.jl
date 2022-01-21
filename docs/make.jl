using LowRankIntegrators
using Documenter

DocMeta.setdocmeta!(LowRankIntegrators, :DocTestSetup, :(using LowRankIntegrators); recursive=true)

makedocs(;
    modules=[LowRankIntegrators],
    authors="Flemming Holtorf",
    repo="https://github.com/FHoltorf/LowRankIntegrators.jl/blob/{commit}{path}#{line}",
    sitename="LowRankIntegrators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FHoltorf.github.io/LowRankIntegrators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FHoltorf/LowRankIntegrators.jl",
    devbranch="main",
)
