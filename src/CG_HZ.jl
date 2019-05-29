export CG_HZ

function CG_HZ(nlp :: AbstractNLPModel,
               stp :: NLPStopping;
               verbose :: Bool=false,
               verboseLS :: Bool = false,
               linesearch :: Function = armijo_ls,
               scaling :: Bool = true,
               kwargs...)

    return CG_generic(nlp,
                      stp;
                      verbose = verbose,
                      verboseLS = verboseLS,
                      linesearch  = linesearch,
                      CG_formula  = formula_HZ,
                      scaling  = scaling,
                      kwargs...)
end
