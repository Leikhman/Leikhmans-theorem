import time
import math
from array import array
import multiprocessing as mp
from multiprocessing import Manager
import sys


def compute_phi(limit):
    """Computes phi(n) (Euler's totient function)."""
    phi = array('I', [0]) * (limit + 1)
    primes = []
    is_composite = bytearray(limit + 1)
    phi[0] = 0
    if limit >= 1:
        phi[1] = 1
    for i in range(2, limit + 1):
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            if i * p > limit:
                break
            is_composite[i * p] = 1
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
    return phi


def divisors(n):
    """All divisors of a number n."""
    n = abs(n)
    divs = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs


def decomposition_rank(delta, omega, n):
    """
    Determines the rank of a decomposition (1 - most interesting, 5 - trivial).
    """
    if (delta == 1 and omega == n) or (delta == n and omega == 1):
        return 5

    coprime = math.gcd(delta, omega) == 1
    product_eq_n = (delta * omega == n)

    if not coprime and not product_eq_n:
        return 1
    if not coprime and product_eq_n:
        return 2
    if coprime and not product_eq_n:
        return 3
    if coprime and product_eq_n:
        return 4
    return 5


def worker(start, end, phi_cache, progress_queue, worker_id, verbose):
    """Process numbers from start to end (inclusive). Reports progress."""
    type_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    errors = []
    total = end - start + 1
    last_report = 0
    for idx, n in enumerate(range(start, end + 1)):
        target = phi_cache[n]
        divs = divisors(n)
        types_found = set()

        if phi_cache[1] * phi_cache[n] == target:
            types_found.add(5)

        for i, delta in enumerate(divs):
            for omega in divs[i:]:
                if phi_cache[delta] * phi_cache[omega] == target:
                    rank = decomposition_rank(delta, omega, n)
                    types_found.add(rank)

        if types_found:
            for rank in types_found:
                type_count[rank] += 1
        else:
            errors.append(n)

        if verbose:
            print(f"Worker {worker_id}: n={n:4d}, phi(n)={target}, types={sorted(types_found)}")

        progress = int((idx + 1) / total * 100)
        if progress > last_report:
            last_report = progress
            progress_queue.put((worker_id, progress, n))

    progress_queue.put((worker_id, 100, end))
    return type_count, errors


def leikhmans_phi_theorem(max_n=10000000, limit=20000000, num_workers=None, verbose=False):
    start_time = time.time()

    print("Computing phi(n) using a linear sieve...")
    phi_cache = compute_phi(limit)
    print(f"Done. Time: {time.time() - start_time:.2f} sec")

    if num_workers is None:
        num_workers = mp.cpu_count()
    print(f"Using {num_workers} workers")

    chunk_size = (max_n + num_workers - 1) // num_workers
    ranges = []
    for i in range(num_workers):
        s = i * chunk_size + 1
        e = min((i + 1) * chunk_size, max_n)
        if s <= e:
            ranges.append((s, e, i))

    manager = Manager()
    progress_queue = manager.Queue()
    completed_workers = 0
    worker_progress = [0] * len(ranges)

    args = [(s, e, phi_cache, progress_queue, wid, verbose) for (s, e, wid) in ranges]

    def progress_reporter():
        nonlocal completed_workers
        while completed_workers < len(ranges):
            try:
                wid, prog, current_n = progress_queue.get(timeout=1)
                worker_progress[wid] = prog
                if prog == 100:
                    completed_workers += 1
                sys.stdout.write(f"\rWorkers: [")
                for p in worker_progress:
                    sys.stdout.write(f"{p:3d}% ")
                sys.stdout.write(f"] Last n: {current_n:<10}")
                sys.stdout.flush()
            except:
                pass

    import threading
    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(worker, args)

    reporter.join(timeout=2)

    total_type_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    all_errors = []
    for type_count, errors in results:
        for k in total_type_count:
            total_type_count[k] += type_count[k]
        all_errors.extend(errors)

    elapsed_time = time.time() - start_time
    # Вывод в консоль
    print("\n" + "=" * 60)
    print("Verification results for phi(n)")
    print("=" * 60)
    print(f"Execution time:                         {elapsed_time:.2f} sec")
    print(f"Total n checked:                        {max_n}")
    print(f"n having type (1):                       {total_type_count[1]:8}")
    print(f"n having type (2):                       {total_type_count[2]:8}")
    print(f"n having type (3):                       {total_type_count[3]:8}")
    print(f"n having type (4):                       {total_type_count[4]:8}")
    print(f"n having type (5):                       {total_type_count[5]:8}")
    print(f"Errors:                                  {len(all_errors):8}")

    if all_errors:
        print("\nFailure, the theorem is not confirmed!")
        print(f"First 10 errors: n = {all_errors[:10]}")
    else:
        print("\nTriumph, the theorem holds for all n checked!")

    # Сохраняем результаты в файл
    with open("results_phi.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Verification results for phi(n)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Execution time:                         {elapsed_time:.2f} sec\n")
        f.write(f"Total n checked:                        {max_n}\n")
        f.write(f"n having type (1):                       {total_type_count[1]:8}\n")
        f.write(f"n having type (2):                       {total_type_count[2]:8}\n")
        f.write(f"n having type (3):                       {total_type_count[3]:8}\n")
        f.write(f"n having type (4):                       {total_type_count[4]:8}\n")
        f.write(f"n having type (5):                       {total_type_count[5]:8}\n")
        f.write(f"Errors:                                  {len(all_errors):8}\n")
        if all_errors:
            f.write("\nFailure, the theorem is not confirmed!\n")
            f.write(f"First 10 errors: n = {all_errors[:10]}\n")
        else:
            f.write("\nTriumph, the theorem holds for all n checked!\n")


if __name__ == "__main__":
    leikhmans_phi_theorem(max_n=100, limit=200, verbose=True)