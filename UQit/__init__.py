from .analyticTestFuncs import fEx1D,fEx2D,fEx3D
from .gpr_torch import gpr, gprPost, gprPlot
from .lagInt import lagInt, lagInt_Quads2Line
from .linAlg import myLinearRegress
from .nodes import Clenshaw_pts,ClenshawCurtis_pts,gllPts
from .pce import pce, pceEval, convPlot
from .ppce import ppce
from .reshaper import lengthVector,vecs2grid,vecsGlue
from .sampling import trainSample,testSample,LHS_sampling
from .sobol import sobol
from .statsUqit import pdfFit_uniVar,pdfPredict_uniVar
from .surr2surr import lagIntAtGQs
from .writeUQ import printRepeated
