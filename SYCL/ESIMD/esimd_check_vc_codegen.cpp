//==-------- esimd_check_vc_codegen.cpp - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// esimd_emulator does not support online-compiler that invokes 'piProgramBuild'
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

int main(void) {
  try {
    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e =
        q.submit([&](handler &cgh) { cgh.single_task<class Test>([] {}); });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    return 1;
  }
  return 0;
}

// CHECK: ---> piProgramBuild(
// CHECK: <const char *>: {{.*}}-vc-codegen
// CHECK: ) ---> pi_result : PI_SUCCESS
