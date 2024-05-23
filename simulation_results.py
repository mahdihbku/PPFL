# -*- coding: utf-8 -*-
"""
Created on May 23 2024

@author: Mahdi 
"""

import secrets
import ecvrf_edwards25519_sha512_elligator2
import time
import os
import concurrent.futures

execution_times = 2

def count_leading_zeros(binary_data):
    # Convert the bytes object to a binary string representation
    binary_string = ''.join(f'{byte:08b}' for byte in binary_data)
    
    # Count the leading zeros in the binary string
    count = 0
    for char in binary_string:
        if char == '0':
            count += 1
        else:
            break
    return count

def perform_verifications(secret_key, public_key, size, num_verifs, execution_times):
    verf_t = 0.0
    for _ in range(execution_times):
        for _ in range(num_verifs):
            selected = 0
            alpha_string = os.urandom(size)
            p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
            b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
            t1 = time.perf_counter()
            result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
            if not (result == "VALID" and beta_string == beta_string2):
                raise Exception("Commitment not verified")
            if count_leading_zeros(beta_string) >= 5:
                selected += 1
            verf_t += time.perf_counter() - t1
    return (size, num_verifs, verf_t / execution_times * 1000)

def parallel_verification(secret_key, public_key, input_byte_size, num_verifications, execution_times):
    times = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(perform_verifications, secret_key, public_key, size, num_verifs, execution_times)
            for size in input_byte_size
            for num_verifs in num_verifications
        ]
        for future in concurrent.futures.as_completed(futures):
            times.append(future.result())

    print("Parallel verification times for input size:")
    print(times)

if __name__ == "__main__":
    # key generation time
    gen_t = 0.0
    for _ in range(execution_times):
        t1 = time.perf_counter()
        secret_key = secrets.token_bytes(nbytes=32)
        public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)
        t2 = time.perf_counter()
        gen_t += t2 - t1
    print("Key generation time: {} ms".format(gen_t/execution_times*1000))

    # proving time x=alpha_string_size
    # input_byte_size = [100, 200, 500, 1000, 2000, 5000, 10000]
    input_byte_size = [100, 200]

    times = []
    for size in input_byte_size:
        prov_t = 0.0
        selected = 0
        for _ in range(execution_times):
            alpha_string = os.urandom(size)
            t1 = time.perf_counter()
            p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
            b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
            if count_leading_zeros(beta_string) >= 5:
                selected += 1
            prov_t += time.perf_counter() - t1
        times.append(prov_t/execution_times*1000)
    print("Proving times for input size:")
    print(input_byte_size)
    print(times)

    # verification time x=alpha_string_size
    times = []
    for size in input_byte_size:
        verf_t = 0.0
        selected = 0
        for _ in range(execution_times):
            alpha_string = os.urandom(size)
            p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
            b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
            t1 = time.perf_counter()
            result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
            if not (result == "VALID" and beta_string == beta_string2):
                raise("Commitment not verified")
            if count_leading_zeros(beta_string) >= 5:
                selected += 1
            verf_t += time.perf_counter() - t1
        times.append(verf_t/execution_times*1000)
    print("Verification times for input size:")
    print(input_byte_size)
    print(times)

    # verification time x=number_of_verifications
    num_verifications = [5, 10]
    input_byte_size = [1000, 2000]
    times = []
    for size in input_byte_size:
        for num_verifs in num_verifications:
            verf_t = 0.0
            for _ in range(execution_times):
                for _ in range(num_verifs):
                    selected = 0
                    alpha_string = os.urandom(size)
                    p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
                    b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
                    t1 = time.perf_counter()
                    result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
                    if not (result == "VALID" and beta_string == beta_string2):
                        raise("Commitment not verified")
                    if count_leading_zeros(beta_string) >= 5:
                        selected += 1
                    verf_t += time.perf_counter() - t1
            times.append(verf_t/execution_times*1000)
    print("Sequential verification times for input size:")
    print([(x,y) for x in input_byte_size for y in num_verifications])
    print(times)

    # same verification in parallel
    parallel_verification(secret_key, public_key, input_byte_size, num_verifications, execution_times)
