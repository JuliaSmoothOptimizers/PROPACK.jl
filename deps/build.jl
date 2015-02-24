p = pwd()
cd(Pkg.dir("PROPACK/deps/PROPACK"))
run(`make`)
cd(p)