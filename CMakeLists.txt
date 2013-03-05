cmake_minimum_required (VERSION 2.8)

# Project Name
project( phdvslam )

# Find package
set(OpenCV_DIR ${OpenCV_DIR} "/opt/local/lib/cmake")
find_package( OpenCV REQUIRED )

# Add executable
add_executable( test src/test.cpp )
include_directories( "src" )
#add_library( cv_static STATIC ${OpenCV_LIBS} )

# Target
target_link_libraries( test ${OpenCV_LIBS} )
#target_link_libraries( test cv_static )
