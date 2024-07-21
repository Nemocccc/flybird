cmake_minimum_required(VERSION 3.10)

project(flybird)

file(READ "version.txt" VERSION)

aux_source_directory(./src DIR_SRCS)

add_executable(flybird ${DIR_SRCS})