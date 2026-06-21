# FindGlog.cmake — provides glog::glog imported target
# for Ubuntu/Debian systems where libgoogle-glog-dev does not ship cmake configs.

find_path(GLOG_INCLUDE_DIR
  NAMES glog/logging.h
  PATHS /usr/include /usr/local/include
)

find_library(GLOG_LIBRARY
  NAMES glog
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glog
  REQUIRED_VARS GLOG_LIBRARY GLOG_INCLUDE_DIR
)

if(Glog_FOUND AND NOT TARGET glog::glog)
  add_library(glog::glog UNKNOWN IMPORTED)
  set_target_properties(glog::glog PROPERTIES
    IMPORTED_LOCATION "${GLOG_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLOG_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(GLOG_INCLUDE_DIR GLOG_LIBRARY)
