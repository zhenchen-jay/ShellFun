include(FetchContent)

# Fetch libshell
FetchContent_Declare(
  libshell
  GIT_REPOSITORY https://github.com/evouga/libshell.git
  GIT_TAG 855a07b7955e3f0311230b17daa805e8ddbb7464 
)

# Make libshell available
FetchContent_MakeAvailable(libshell)