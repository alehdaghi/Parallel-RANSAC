#cmake_minimum_required(VERSION 2.8)
#project( opencv-wrapper-egbis )
#find_package( OpenCV REQUIRED )
#add_executable( egbis segment.cpp segment-image.h segment-graph.h disjoint-set.h image.h )
#target_link_libraries( egbis ${OpenCV_LIBS} )
#set(SOURCE
#   ${SOURCE}
#   ${CMAKE_CURRENT_SOURCE_DIR}/segment.cpp
#   PARENT_SCOPE
#)
set(SOURCE
   ${SOURCE}
   ${CMAKE_CURRENT_SOURCE_DIR}/convolve.h
   ${CMAKE_CURRENT_SOURCE_DIR}/filter.h
   ${CMAKE_CURRENT_SOURCE_DIR}/imconv.h
   ${CMAKE_CURRENT_SOURCE_DIR}/misc.h
   ${CMAKE_CURRENT_SOURCE_DIR}/segment-graph.h
   ${CMAKE_CURRENT_SOURCE_DIR}/disjoint-set.h
   ${CMAKE_CURRENT_SOURCE_DIR}/image.h
   ${CMAKE_CURRENT_SOURCE_DIR}/imutil.h
   ${CMAKE_CURRENT_SOURCE_DIR}/pnmfile.h
   ${CMAKE_CURRENT_SOURCE_DIR}/segment-image.h
   PARENT_SCOPE
)
set(HEADERS
   ${HEADERS}
   ${CMAKE_CURRENT_SOURCE_DIR}/convolve.h  
   ${CMAKE_CURRENT_SOURCE_DIR}/filter.h
   ${CMAKE_CURRENT_SOURCE_DIR}/imconv.h
   ${CMAKE_CURRENT_SOURCE_DIR}/misc.h
   ${CMAKE_CURRENT_SOURCE_DIR}/segment-graph.h
   ${CMAKE_CURRENT_SOURCE_DIR}/disjoint-set.h
   ${CMAKE_CURRENT_SOURCE_DIR}/image.h
   ${CMAKE_CURRENT_SOURCE_DIR}/imutil.h
   ${CMAKE_CURRENT_SOURCE_DIR}/pnmfile.h
   ${CMAKE_CURRENT_SOURCE_DIR}/segment-image.h
   PARENT_SCOPE
)

