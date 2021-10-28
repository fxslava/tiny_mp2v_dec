project(tiny_mp2v_dec)

set(CMAKE_CXX_STANDARD 11)

file(
    GLOB INC_DIRECTORIES
        ./api
)

include_directories(
    ${INC_DIRECTORIES}
)

file(
    GLOB SOURCES
        ./core/decoder.cpp
        ./core/mb_decoder.cpp
        ./core/mp2v_vlc.cpp
        ./core/mp2v_hdr.cpp
        ./core/scan_c.cpp
        ./core/threads.cpp
        ./core/mc.cpp
)

if(WIN32)
    set(
        CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} /fp:precise"
    )
elseif(UNIX)
    set(
        CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Ofast -g0 -Wall"
    )
endif()

add_library(
	${PROJECT_NAME}
	${SOURCES}
	)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER INC_DIRECTORIES)
target_include_directories(${PROJECT_NAME} PRIVATE .)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "libs")
