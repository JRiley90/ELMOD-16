from elmod16.v2_engine import evolve_v2

if __name__ == "__main__":
    best, score = evolve_v2(pop_size=20, generations=15, verbose=True)
    print("\nBest V2 score:", score)
