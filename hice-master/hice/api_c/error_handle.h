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
 * This file is based on src\c_api\c_api_error.h from MXNET.
 * And it's slightly modified for HICE's usage.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#pragma once

#include <string>
#include "hice/core/status.h"
#include "hice/util/loguru.h"

/*!
 * \brief Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END() or API_END_HANDLE_ERROR()
 * The finally clause contains procedure to cleanup states when an error
 * happens.
 */
#define HI_API_BEGIN() \
  try {                \
    hice::on_enter_api(__FUNCTION__);

#define HI_API_END()                 \
  }                                  \
  catch (const hice::Exception &e) { \
    hice::on_exit_api();             \
    hice::HandleException(e);        \
    return (HI_Status)(e.code());    \
  }                                  \
  hice::on_exit_api();               \
  return HI_Status::Success;

#define HI_API_END_HANDLE_ERROR(Finalize) \
  }                                       \
  catch (const hice::Exception &e) {      \
    Finalize;                             \
    hice::on_exit_api();                  \
    hice::HandleException(e);             \
    return (HI_Status)(e.code()));        \
  }                                       \
  hice::on_exit_api();                    \
  return HI_Status::Success;

namespace hice {

class Exception : public std::exception {
 public:
  explicit Exception(const std::string& msg, Status code) : msg_(msg), code_(code) {}
  const char* what() const noexcept override { return msg_.c_str(); }
  Status code() const noexcept { return code_; }

 private:
  std::string msg_;
  Status code_;
};

/*!
 * \brief Parse and process error message, then throw an exception
 * \param user_data unused.
 * \param message The error message from loguru.
 *                
 */
void PrepareException(const loguru::Message &message);

/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void SetLastError(const char *msg);

/*!
 * \brief Get the last error message
 * \return msg The error message.
 */
const char *GetLastError();

/*!
 * \brief handle exception throwed out
 * \param e the exception
 */
void HandleException(const Exception &e);

void on_enter_api(const char *function);
void on_exit_api();

}  // namespace hice
