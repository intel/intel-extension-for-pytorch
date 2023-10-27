#include "xpu_output_json.h"

#include <fmt/format.h>
#include <time.h>
#include <fstream>
#include <map>

#include "kineto/Config.h"
#include "kineto/TraceSpan.h"

#include "Logger.h"

namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
static constexpr char kFlowStart = 's';
static constexpr char kFlowEnd = 'f';

#ifdef __linux__
static constexpr char kDefaultLogFileFmt[] =
    "/tmp/libkineto_activities_{}.json";
#else
static constexpr char kDefaultLogFileFmt[] = "libkineto_activities_{}.json";
#endif

std::string& ChromeTraceLogger::sanitizeStrForJSON(std::string& value) {
// Replace all backslashes with forward slash because Windows paths causing
// JSONDecodeError.
#ifdef _WIN32
  std::replace(value.begin(), value.end(), '\\', '/');
#endif
  return value;
}

void ChromeTraceLogger::metadataToJSON(
    const std::unordered_map<std::string, std::string>& metadata) {
  for (const auto& kv : metadata) {
    traceOf_ << fmt::format(
        R"JSON(
  "{}": {},)JSON",
        kv.first,
        kv.second);
  }
}

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata) {
  traceOf_ << fmt::format(
      R"JSON(
{{
  "schemaVersion": {},)JSON",
      kSchemaVersion);

  metadataToJSON(metadata);
  traceOf_ << R"JSON(
  "traceEvents": [)JSON";
}

static std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

void ChromeTraceLogger::openTraceFile() {
  tempFileName_ = fileName_ + ".tmp";
  traceOf_.open(tempFileName_, std::ofstream::out | std::ofstream::trunc);
  if (!traceOf_) {
    PLOG(ERROR) << "Failed to open '" << fileName_ << "'";
  } else {
    LOG(INFO) << "Tracing to temporary file " << fileName_;
  }
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName) {
  fileName_ = traceFileName.empty() ? defaultFileName() : traceFileName;
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile();
}

void ChromeTraceLogger::handleDeviceInfo(
    const DeviceInfo& info,
    uint64_t time) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_labels", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "labels": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}, "pid": {}, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time, info.id,
      info.name,
      time, info.id,
      info.label,
      time, info.id,
      info.id < 8 ? info.id + 0x1000000ll : info.id);
  // clang-format on
}

void ChromeTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}, "pid": {}, "tid": {},
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "thread_sort_index", "ph": "M", "ts": {}, "pid": {}, "tid": {},
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time, info.deviceId, info.id,
      info.name,
      time, info.deviceId, info.id,
      info.sortIndex);
  // clang-format on
}

void ChromeTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}, "pid": -1, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}, "pid": -1, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time,
      info.name,
      time,
      0x100000All);
  // clang-format on
}

void ChromeTraceLogger::handleTraceSpan(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Trace", "ts": {}, "dur": {},
    "pid": "Spans", "tid": "{}",
    "name": "{}{} ({})",
    "args": {{
      "Op count": {}
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {},
    "pid": "Spans", "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      span.startTime, span.endTime - span.startTime,
      span.name,
      span.prefix, span.name, span.iteration,
      span.opCount,
      span.startTime,
      0x20000000ll);
  // clang-format on

  addIterationMarker(span);
}

void ChromeTraceLogger::addIterationMarker(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Iteration Start: {}", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace {}", "ts": {}
  }},)JSON",
      span.name,
      span.name, span.startTime);
  // clang-format on
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  traceOf_ << fmt::format(
      R"JSON(
  {{
    "ph": "i", "cat": "{}", "s": "t", "name": "{}",
    "pid": {}, "tid": {},
    "ts": {},
    "args": {{
      {}
    }}
  }},)JSON",
      toString(op.type()),
      op.name(),
      op.deviceId(),
      op.resourceId(),
      op.timestamp(),
      op.metadataJson());
}

void ChromeTraceLogger::handleActivity(const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  int64_t ts = op.timestamp();
  int64_t duration = op.duration();

  if (duration < 0) {
    duration = 0;
  }

  if (op.type() == ActivityType::GPU_USER_ANNOTATION) {
    ts--;
    duration++; // Still need it to end at the orginal point
  }

  std::string arg_values = "";
  if (op.correlationId() != 0) {
    arg_values.append(fmt::format("\"External id\": {}", op.correlationId()));
  }
  const std::string op_metadata = op.metadataJson();
  if (op_metadata.find_first_not_of(" \t\n") != std::string::npos) {
    if (!arg_values.empty()) {
      arg_values.append(",");
    }
    arg_values.append(op_metadata);
  }
  std::string args = "";
  if (!arg_values.empty()) {
    args = fmt::format(
        R"JSON(,
    "args": {{
      {}
    }})JSON",
        arg_values);
  }

  int device = op.deviceId();
  int resource = op.resourceId();
  std::string op_name = op.name() == "kernel" ? "Kernel" : op.name();

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "{}", "name": "{}", "pid": {}, "tid": {},
    "ts": {}, "dur": {}{}
  }},)JSON",
          toString(op.type()), op_name, device, resource,
          ts, duration, args);
  // clang-format on
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
  handleActivity(op);
}

void ChromeTraceLogger::handleGenericLink(const ITraceActivity& act) {
  static struct {
    int type;
    char name[16];
  } flow_names[] = {{kLinkFwdBwd, "fwdbwd"}, {kLinkAsyncCpuGpu, "ac2g"}};
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.name);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.name);
      }
      return;
    }
  }
  LOG(WARNING) << "Unknown flow type: " << act.flowType();
}

void ChromeTraceLogger::handleLink(
    char type,
    const ITraceActivity& e,
    int64_t id,
    const std::string& name) {
  if (!traceOf_) {
    return;
  }

  const auto binding = (type == kFlowEnd) ? ", \"bp\": \"e\"" : "";
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "{}", "id": {}, "pid": {}, "tid": {}, "ts": {},
    "cat": "{}", "name": "{}"{}
  }},)JSON",
      type, id, e.deviceId(), e.resourceId(), e.timestamp(), name, name, binding);
  // clang-format on
}

void ChromeTraceLogger::finalizeTrace(
    const Config& /*unused*/,
    std::unique_ptr<XPUActivityBuffers> /*unused*/,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  LOG(INFO) << "Chrome Trace written to " << fileName_;
  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": {}
  }}
  ],)JSON",
      endTime);

#if !USE_GOOGLE_LOG
  std::unordered_map<std::string, std::string> PreparedMetadata;
  for (const auto& kv : metadata) {
    if (!kv.second.empty()) {
      std::string value = "[";
      int mdv_count = kv.second.size();
      for (const auto& v : kv.second) {
        value.append("\"" + v + "\"");
        if(mdv_count > 1) {
          value.append(",");
          mdv_count--;
        }
      }
      value.append("]");
      PreparedMetadata[kv.first] = sanitizeStrForJSON(value);
    }
  }
  metadataToJSON(PreparedMetadata);
#endif // !USE_GOOGLE_LOG

  traceOf_ << fmt::format(R"JSON(
  "traceName": "{}"
}})JSON", sanitizeStrForJSON(fileName_));
  // clang-format on

  traceOf_.close();
  // On some systems, rename() fails if the destination file exists.
  // So, remove the destination file first.
  remove(fileName_.c_str());
  if (rename(tempFileName_.c_str(), fileName_.c_str()) != 0) {
    PLOG(ERROR) << "Failed to rename " << tempFileName_ << " to " << fileName_;
  } else {
    LOG(INFO) << "Renamed the trace file to " << fileName_;
  }
}

} // namespace KINETO_NAMESPACE
