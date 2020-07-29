from ftis.analyser import FluidMFCC, Stats
from ftis.process import FTISProcess

proc = FTISProcess("Testing Hits", "anal")
proc.add(
    FluidMFCC(numcoeffs=20),
    Stats(numderivs=1)
)

if __name__ == "__main__":
    proc.run()