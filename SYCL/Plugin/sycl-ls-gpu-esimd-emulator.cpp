// REQUIRES: gpu, esimd_emulator

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}gpu, {{.*}}ESIMD_EMULATOR
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}gpu, {{.*}}ESIMD_EMULATOR

//= sycl-ls-gpu-esimd-emulator.cpp - Test ESIMD_EMULATOR selected gpu device =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
