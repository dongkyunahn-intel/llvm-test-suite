/* Toy test for reproducing double-conversion of half type under
   ESIMD_EMULATOR backend

   Reference for half-type conversion : https://evanw.github.io/float-toy/
 */
// REQUIRES: gpu && esimd_emulator
// RUN: %clangxx -fsycl -g %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <stdlib.h>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace cl::sycl::ext;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <typename Ty> struct test_id;

template <class Ty> bool test(queue q, int inc){
  Ty *data = new Ty[1];

  data[0] = (Ty)0;
  Ty VAL = (Ty)inc;

  try {
    buffer<Ty, 1> buf(data, range<1>(1));
    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.single_task<test_id<Ty>>([=]() SYCL_ESIMD_KERNEL {
        simd<uint32_t, 1> offsets(0);
        simd<Ty, 1> vec = gather<Ty,1>(data,offsets);
        vec[0] += (Ty)inc;
        /// For 'half' type, float2Half is called unnecessarily for
        /// 2nd argument of 'scalar_store' which is already correct in
        /// half-type value
        scalar_store<Ty>(acc,0,vec[0]);
      });
    });
  }
  catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] data;
    return false;
  }

  using Tint = esimd_test::int_type_t<sizeof(Ty)>;
  Tint ResBits = *(Tint *)&data[0];
  Tint GoldBits = *(Tint *)&VAL;

  std::cout << "Comparison of representation '" << inc << "' of Type "
            << typeid(Ty).name() << std::endl;
  std::cout << "Bits(data[0]) = 0x" << std::hex << ResBits << " / "
            << "Bits(GOLD) = 0x" << GoldBits << std::dec <<std::endl;

  if (VAL == data[0]) {
    std::cout << "Pass";
  }
  else {
    std::cout << "Fail";
  }
  
  // std::cout << "for Typeid = " << typeid(Ty).name() << std::endl;
  return ((Ty)inc == data[0]);
}


int main(int argc, char *argv[]) {
  bool passed = true;
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  // std::cout << "\n===================" << std::endl;
  // passed &= test<short>(q, 1);
  std::cout << "\n===================" << std::endl;
  passed &= test<half>(q, 1);
  std::cout << "\n===================" << std::endl;
  passed &= test<float>(q, 1);
  std::cout << "\n===================" << std::endl;

  if (passed) {
    std::cout << "Pass!!" << std::endl;
  }
  else {
    std::cout << "Fail!!" << std::endl;
  }

  return passed ? 0 : -1;
}
