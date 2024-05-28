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

execution_times = 4

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

# def perform_verifications(secret_key, public_key, size, num_verifs, execution_times):
#     verf_t = 0.0
#     for _ in range(execution_times):
#         for _ in range(num_verifs):
#             selected = 0
#             alpha_string = os.urandom(size)
#             p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
#             b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
#             t1 = time.perf_counter()
#             result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
#             if not (result == "VALID" and beta_string == beta_string2):
#                 raise Exception("Commitment not verified")
#             if count_leading_zeros(beta_string) >= 5:   # just a random test to use beta_string
#                 selected += 1
#             verf_t += time.perf_counter() - t1
#     return (size, num_verifs, verf_t / execution_times * 1000)

# def parallel_verification(secret_key, public_key, input_byte_size, num_verifications, execution_times):
#     times = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         futures = [
#             executor.submit(perform_verifications, secret_key, public_key, size, num_verifs, execution_times)
#             for size in input_byte_size
#             for num_verifs in num_verifications
#         ]
#         for future in concurrent.futures.as_completed(futures):
#             times.append(future.result())

#     print("Parallel verification times for input size:")
#     print(times)

def verify_single_instance(public_key, alpha_string, pi_string, beta_string):
    selected = 0
    result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
    if not (result == "VALID" and beta_string == beta_string2):
        raise Exception("Commitment not verified")
    if count_leading_zeros(beta_string) >= 5:
        selected += 1
    return selected

def parallel_verification(secret_key, public_key, input_byte_size, num_verifications):
    times_list = []
    for _ in range(execution_times):
        times = []
        for size in input_byte_size:
            for num_verifs in num_verifications:
                alpha_list = []
                pi_list = []
                beta_list = []
                for _ in range(num_verifs):
                    alpha_string = os.urandom(size)
                    _, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
                    _, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
                    alpha_list.append(alpha_string)
                    pi_list.append(pi_string)
                    beta_list.append(beta_string)
                t1 = time.perf_counter()
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(verify_single_instance, public_key, alpha_list[i], pi_list[i], beta_list[i])
                        for i in range(num_verifs)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        selected = future.result()
                times.append((time.perf_counter() - t1)  * 1000)
        times_list.append(times)

    
    print("Verification times for input size and number of verifications:")
    combinations_list = [(x, y) for x in input_byte_size for y in num_verifications]
    print(combinations_list)
    avg_times = [sum(x) / len(x) for x in zip(*times_list)]
    print(avg_times)

    f = open("experiments/vrf_verif_time.csv", "w")
    for i in range(len(combinations_list)):
        f.write(str(combinations_list[i])+', '+str(avg_times[i])+'\n')
    f.close()

if __name__ == "__main__":
    input_byte_size = [100, 200, 500, 1000, 2000, 5000, 10000, 100000]
    # key generation time
    key_gen_times = []
    for _ in range(len(input_byte_size)):
        gen_t = 0.0
        for _ in range(execution_times):
            t1 = time.perf_counter()
            secret_key = secrets.token_bytes(nbytes=32)
            public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)
            t2 = time.perf_counter()
            gen_t += t2 - t1
        print("Key generation time: {} ms".format(gen_t/execution_times*1000))
        key_gen_times.append(gen_t/execution_times*1000)

    # # proving time x=alpha_string_size
    # prov_times = []
    # for size in input_byte_size:
    #     prov_t = 0.0
    #     selected = 0
    #     for _ in range(execution_times):
    #         alpha_string = os.urandom(size)
    #         t1 = time.perf_counter()
    #         p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
    #         b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    #         if count_leading_zeros(beta_string) >= 5:
    #             selected += 1
    #         prov_t += time.perf_counter() - t1
    #     prov_times.append(prov_t/execution_times*1000)
    # print("Proving times for input size:")
    # print(input_byte_size)
    # print(prov_times)

    # # verification time x=alpha_string_size
    # times_list = []
    # for _ in range(execution_times):
    #     times = []
    #     for size in input_byte_size:
    #         verf_t = 0.0
    #         selected = 0
    #         alpha_string = os.urandom(size)
    #         p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
    #         b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    #         t1 = time.perf_counter()
    #         result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
    #         if not (result == "VALID" and beta_string == beta_string2):
    #             raise("Commitment not verified")
    #         if count_leading_zeros(beta_string) >= 5:
    #             selected += 1
    #         times.append((time.perf_counter() - t1)*1000)
    #     times_list.append(times)
    # print("Verification times for input size:")
    # print(input_byte_size)
    # avg_times = [sum(x) / len(x) for x in zip(*times_list)]
    # print([sum(x) / len(x) for x in zip(*times_list)])

    # f = open("experiments/vrf_keygen_prove_time.csv", "w")
    # for i in range(len(input_byte_size)):
    #     f.write(str(input_byte_size[i])+', '+str(key_gen_times[i])+', '+str(prov_times[i])+', '+str(avg_times[i])+'\n')
    # f.close()

    # # verification time x=number_of_verifications
    num_verifications = [1, 5, 10, 15, 20, 25, 30]
    input_byte_size = [1000]
    # times = []
    # for size in input_byte_size:
    #     for num_verifs in num_verifications:
    #         verf_t = 0.0
    #         for _ in range(execution_times):
    #             for _ in range(num_verifs):
    #                 selected = 0
    #                 alpha_string = os.urandom(size)
    #                 p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
    #                 b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    #                 t1 = time.perf_counter()
    #                 result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
    #                 if not (result == "VALID" and beta_string == beta_string2):
    #                     raise("Commitment not verified")
    #                 if count_leading_zeros(beta_string) >= 5:
    #                     selected += 1
    #                 verf_t += time.perf_counter() - t1
    #         times.append(verf_t/execution_times*1000)
    # print("Sequential verification times for input size:")
    # print([(x,y) for x in input_byte_size for y in num_verifications])
    # print(times)

    # times = []
    # for size in input_byte_size:
    #     for num_verifs in num_verifications:
    #         verf_t = 0.0
    #         for _ in range(num_verifs):
    #             selected = 0
    #             alpha_string = os.urandom(size)
    #             p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
    #             b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    #             t1 = time.perf_counter()
    #             result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
    #             if not (result == "VALID" and beta_string == beta_string2):
    #                 raise("Commitment not verified")
    #             if count_leading_zeros(beta_string) >= 5:
    #                 selected += 1
    #             verf_t += time.perf_counter() - t1
    #         times.append(verf_t*1000)
    # print("Sequential verification times for input size:")
    # print([(x,y) for x in input_byte_size for y in num_verifications])
    # print(times)

    # same verification in parallel
    parallel_verification(secret_key, public_key, input_byte_size, num_verifications)
