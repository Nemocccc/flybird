cmake_minimum_required(VERSION 3.10)

project(flybird)

file(READ "version.txt" VERSION)

aux_source_directory(./src DIR_SRCS)

add_executable(flybird ${DIR_SRCS})

#build a CPack driven installer package
include(InstallRequiredSystemLibraries)
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION "${VERSION}")
set(CPACK_PACKAGE_NAME "flybird")
set(CPACK_PACKAGE_CONTACT "Nemo 1774747097@qq.com")
include(CPack)