import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
faiss.omp_set_num_threads(1)

from app.controllers.benchMarkController import runBenchmark
from app.views.cliView import print_table

def main():
    runBenchmark(viewPrint = print_table)

if __name__ == "__main__":
    main()
