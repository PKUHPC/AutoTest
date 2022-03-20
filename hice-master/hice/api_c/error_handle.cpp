/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * This file is based on src\c_api\c_api_error.cc from MXNET.
 * And it's slightly modified for HICE's usage.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hice/api_c/error_handle.h"

#include "hice/util/thread_local.h"

namespace hice {

/*!
 * \brief Parse error message and form as following:
 * Hice::{XxxError} Ocurred.
 * Stack trace:
 *      ...
 *   2  ...
 *   1  ...
 *   0  ...
 * {Time} {Address} {filename}:{line} {error_prefix} {error_message}
 */
void PrepareException(const loguru::Message &message) {
  int skip = 6;
  using std::string;
  // get stacktrace, preamble, prefix, message
  loguru::Text stc = loguru::stacktrace(skip);
  const char *pmb = message.preamble;
  const char *pfx = message.prefix;
  const char *msg = message.message;

  // make error_header(contains stacktrace, preamble and prefix)
  string error_header =
      string("Stack trace:\n") + stc.c_str() + "\nError Details:\n" + pmb + pfx;

  // parse ErrorType from msg by dividing msg into
  // msg_prefix + "{" + Hice::ErrorType + "}" + msg_suffix
  std::istringstream is(msg);
  string msg_prefix, msg_err_type, msg_suffix;
  string error_type;
  hice::Status status;
  bool format_match = getline(is, msg_prefix, '{') &&
                      getline(is, msg_err_type, '}') &&
                      (msg_err_type.compare(0, 6, "Hice::") == 0);
  if (!format_match) {
    // error_preheader contains "XXXError Occurred"
    string error_preheader = "Hice::UnknownError Occurred.\n";
    throw Exception(error_preheader + error_header + msg, hice::kUnknownError);
  }
  error_type = msg_err_type.substr(7);
  while (is.peek() == ' ') is.get();
  getline(is, msg_suffix);

  // get ErrorType
  status = hice::str2status(error_type);
  string error_preheader = string("Hice::") + error_type + " Occurred.\n";
  throw Exception(error_preheader + error_header + msg_prefix + msg_suffix,
                  status);
}  // namespace hice

struct ErrorEntry {
  std::string last_error;
};

typedef ThreadLocalStore<ErrorEntry> APIErrorStore;

void SetLastError(const char *msg) { APIErrorStore::Get()->last_error = msg; }

const char *GetLastError() { return APIErrorStore::Get()->last_error.c_str(); }

void HandleException(const Exception &e) { hice::SetLastError(e.what()); }

void on_enter_api(const char *function) {
  // do something
}

void on_exit_api() {
  // do something
}

}  // namespace hice