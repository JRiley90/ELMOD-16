from elmod16.v1_engine import evolve_v1

if __name__ == "__main__":
    best, score = evolve_v1(pop_size=30, generations=20, verbose=True)
    print("\nBest V1 score:", score)
