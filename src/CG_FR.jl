export CG_FR

function CG_FR(nlp :: AbstractNLPModel,
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
                      CG_formula  = formula_FR,
                      scaling  = scaling,
                      kwargs...)
end
