$(document).ready(function() {
  var svg_copy = "<svg class=\"svg-copy\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" x=\"0px\" y=\"0px\" viewBox=\"0 0 111.07 122.88\" style=\"enable-background:new 0 0 111.07 122.88; cursor: pointer;\" xml:space=\"preserve\"><style type=\"text/css\"><![CDATA[.st0{fill-rule:evenodd;clip-rule:evenodd;}]]></style><g><path class=\"st0\" d=\"M97.67,20.81L97.67,20.81l0.01,0.02c3.7,0.01,7.04,1.51,9.46,3.93c2.4,2.41,3.9,5.74,3.9,9.42h0.02v0.02v75.28 v0.01h-0.02c-0.01,3.68-1.51,7.03-3.93,9.46c-2.41,2.4-5.74,3.9-9.42,3.9v0.02h-0.02H38.48h-0.01v-0.02 c-3.69-0.01-7.04-1.5-9.46-3.93c-2.4-2.41-3.9-5.74-3.91-9.42H25.1c0-25.96,0-49.34,0-75.3v-0.01h0.02 c0.01-3.69,1.52-7.04,3.94-9.46c2.41-2.4,5.73-3.9,9.42-3.91v-0.02h0.02C58.22,20.81,77.95,20.81,97.67,20.81L97.67,20.81z M0.02,75.38L0,13.39v-0.01h0.02c0.01-3.69,1.52-7.04,3.93-9.46c2.41-2.4,5.74-3.9,9.42-3.91V0h0.02h59.19 c7.69,0,8.9,9.96,0.01,10.16H13.4h-0.02v-0.02c-0.88,0-1.68,0.37-2.27,0.97c-0.59,0.58-0.96,1.4-0.96,2.27h0.02v0.01v3.17 c0,19.61,0,39.21,0,58.81C10.17,83.63,0.02,84.09,0.02,75.38L0.02,75.38z M100.91,109.49V34.2v-0.02h0.02 c0-0.87-0.37-1.68-0.97-2.27c-0.59-0.58-1.4-0.96-2.28-0.96v0.02h-0.01H38.48h-0.02v-0.02c-0.88,0-1.68,0.38-2.27,0.97 c-0.59,0.58-0.96,1.4-0.96,2.27h0.02v0.01v75.28v0.02h-0.02c0,0.88,0.38,1.68,0.97,2.27c0.59,0.59,1.4,0.96,2.27,0.96v-0.02h0.01 h59.19h0.02v0.02c0.87,0,1.68-0.38,2.27-0.97c0.59-0.58,0.96-1.4,0.96-2.27L100.91,109.49L100.91,109.49L100.91,109.49 L100.91,109.49z\"/></g></svg>";
  var svg_done = "<svg version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" x=\"0px\" y=\"0px\" viewBox=\"0 0 122.88 109.76\" style=\"enable-background:new 0 0 122.88 109.76\" xml:space=\"preserve\"><style type=\"text/css\">.st0{fill-rule:evenodd;clip-rule:evenodd;}</style><g><path fill=\"#01A601\" class=\"st0\" d=\"M0,52.88l22.68-0.3c8.76,5.05,16.6,11.59,23.35,19.86C63.49,43.49,83.55,19.77,105.6,0h17.28 C92.05,34.25,66.89,70.92,46.77,109.76C36.01,86.69,20.96,67.27,0,52.88L0,52.88z\"/></g></svg>";

  var search_fields = [];
  if(window.location.hash.startsWith("#installation")) {
    var hash_list = decodeURIComponent(window.location.hash).split("?");
    if(hash_list.length > 1) {
      var data = hash_list[1].split("&");
      $.each(data, function(index, value) {
        var pair = value.split("=");
        if(pair.length == 2) {
          search_fields[pair[0]] = pair[1];
        }
      });
    }
    setTimeout(function () {
      $("#menu-installation").trigger("click");
    }, 10);
  } else {
    setTimeout(function () {
      $("#menu-introduction").trigger("click");
    }, 10);
  }

  $(".menu-element").on("click", function() {
    $(this).parent().children().each(function() {
      $(this).removeClass("selected");
    });
    $(this).addClass("selected");

    if($(this).attr("id") == "menu-installation") {
      $("#div-introduction").hide();
      $("#div-installation").show();
      if(!window.location.hash.startsWith("#installation"))
        window.location.hash = "#installation";
      if($("#col-headings").children().length == 0) {
        var query = {};
        query.request = "platform";
        $.ajax_query(JSON.stringify(query));
      }
    } else if($(this).attr("id") == "menu-introduction") {
      $("#div-introduction").show();
      $("#div-installation").hide();
      if(!window.location.hash.startsWith("#introduction"))
        window.location.hash = "#introduction";
    } else {
      // Do nothing
    }
  });

  $.ul_gen = function(data) {
    var ret = "<ul class\"simple\">";
    $.each(data, function(index, value) {
      ret += "<li><p>" + value + "</p></li>";
    });
    ret += "</ul>";
    return ret;
  }

  $.notes_gen = function(data, header = "Note") {
    var ret = "<div class=\"admonition note\"><p class=\"admonition-title\">" + header + "</p>";
    var ul = data.length > 1;
    if(ul) {
      ret += "<ul>";
    }
    $.each(data, function(index, value) {
      if(ul)
        ret += "<li>";
      ret += "<p>" + value + "</p>";
      if(ul)
        ret += "</li>";
    });
    if(ul) {
      ret += "</ul>";
    }
    ret += "</div>";
    return ret;
  }

  $.table_1d_gen = function(headers, data) {
    var ret = "<table border=\"1\" class=\"docutils\">";
    ret += "<tbody>";
    $.each(headers, function(index, header) {
      ret += "<tr>";
      ret += "<td><strong>" + header + "</strong></td>";
      ret += "<td>" + data[index] + "</td>";
      ret += "</tr>";
    });
    ret += "</tbody>";
    ret += "</table>";
    return ret;
  }

  $.table_2d_gen = function(data, horizontal = true) {
    var headers = [];
    $.each(data, function(index, row) {
      headers = $.merge(headers, Object.keys(row));
    });
    headers = headers.filter(function(item, i, headers) {
      return i == headers.indexOf(item);
    });
    var values = [];
    $.each(data, function(index, row) {
      var value = [];
      $.each(headers, function(index, head) {
        if(row[head] != null)
          value.push(row[head]);
        else
          value.push("-");
      });
      values.push(value);
    });

    var ret = "<table border=\"1\" class=\"docutils\">";
    if(horizontal) {
      ret += "<thead><tr>";
      $.each(headers, function(index, value) {
        ret += "<th>" + value + "</th>";
      });
      ret += "</tr></thead>";

      ret += "<tbody>";
      $.each(values, function(index, row) {
        ret += "<tr>";
        $.each(row, function(index, value) {
          ret += "<td>" + value + "</td>";
        });
        ret += "</tr>";
      });
      ret += "</tbody>";
    } else {
      ret += "<tbody>";
      $.each(headers, function(index_h, header) {
        ret += "<tr>";
        ret += "<td><strong>" + header + "</strong></td>";
        $.each(values, function(index_r, row) {
          ret += "<td>" + row[index_h] + "</td>";
        });
        ret += "</tr>";
      });
      ret += "</tbody>";
    }
    ret += "</table>";
    return ret;
  }

  $.code_gen = function(data) {
    return "<div class=\"code-div\">"
          + "<div class=\"highlight-bash notranslate\"><div class=\"highlight\"><pre>" + data.join("\n") + "</pre></div></div>"
          + "<div class=\"code-svg\">" + svg_copy + "</div>"
          + "</div>";
  }

  $.secid_gen = function(indice_main, index_sub = -1) {
    var ret = indice_main.join(".");
    if(index_sub > -1)
      ret += "[" + String.fromCharCode('a'.charCodeAt(0) + index_sub) + "]";
    ret += ". ";
    return ret;
  }

  $.pkgInArray = function(pkg, list) {
    return $.inArray(pkg, list) > -1;
  }

  $.instruction_gen = function(data) {
    var ret = "";
    var indexa = 1;
    if(data.system_requirements != null) {
      ret += "<div class=\"simple\">";
      ret += "<h2>" + $.secid_gen([indexa]) + "System Requirements</h2>";
      ret += "<p>Make sure that your system complies with the following system requirements:</p>";

      var indexb = 1;
      if(data.system_requirements.hardware != null) {
        ret += "<div class=\"simple\">";
        ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Hardware</h3>";

        if(data.system_requirements.hardware.support != null) { // Array of text
          ret += $.ul_gen(data.system_requirements.hardware.support);
        }
        if(data.system_requirements.hardware.verified != null) { // Array of text
          ret += "<p><strong>Verified on:</strong></p>";
          ret += $.ul_gen(data.system_requirements.hardware.verified);
        }
        if(data.system_requirements.hardware.notes != null) { // Array of text
          ret += $.notes_gen(data.system_requirements.hardware.notes);
        }

        ret += "</div>";
        indexb += 1;
      }
      if(data.system_requirements.software != null) {
        ret += "<div class=\"simple\">";
        ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Software</h3>";

        if(data.system_requirements.software.driver != null) {
          ret += $.table_2d_gen(data.system_requirements.software.driver);
        }

        if(!$.pkgInArray(data.package, ["docker"])) {
          var items = [];
          if(data.system_requirements.software.os != null) {
            var item = {};
            item["Software"] = "OS";
            item["Version"] = data.system_requirements.software.os.join(", ");
            items.push(item);
          }
          if(data.system_requirements.software.basekit != null) {
            var applyto = data.system_requirements.software.basekit.applyto;
            if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
              var item = {};
              item["Software"] = "Intel® oneAPI Base Toolkit";
              item["Version"] = data.system_requirements.software.basekit.version;
              items.push(item);
            }
          }
          if(data.system_requirements.software.basekit_hotfix != null) {
            var applyto = data.system_requirements.software.basekit_hotfix.applyto;
            if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
              var item = {};
              item["Software"] = "Intel® oneAPI Base Toolkit Hotfix";
              item["Version"] = data.system_requirements.software.basekit_hotfix;
              items.push(item);
            }
          }
          if(data.system_requirements.software.pti != null) {
            var applyto = data.system_requirements.software.basekit.applyto;
            if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
              var item = {};
              item["Software"] = "Intel® Profiling Tools Interfaces for GPU (PTI for GPU)";
              item["Version"] = data.system_requirements.software.pti.version;
              items.push(item);
            }
          }
          if(data.system_requirements.software.python != null) {
            var item = {};
            item["Software"] = "Python";
            item["Version"] = data.system_requirements.software.python.join(", ");
            items.push(item);
          }
          if(data.system_requirements.software.compiler != null &&
             $.pkgInArray(data.package, ["source", "cppsdk"])) {
            var item = {};
            item["Software"] = "Compiler";
            item["Version"] = data.system_requirements.software.compiler;
            items.push(item);
          }
          ret += $.table_2d_gen(items);
        }

        ret += "</div>";
        indexb += 1;
      }

      ret += "</div>";
      indexa += 1;
    }
    if(data.preparation != null) {
      ret += "<div class=\"simple\">";
      ret += "<h2>" + $.secid_gen([indexa]) + "Preparation</h2>";

      var indexb = 1;
      if(data.preparation.driver != null) {
        ret += "<div class=\"simple\">";
        ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Driver</h3>";
        ret += $.table_2d_gen(data.preparation.driver);
        ret += "</div>";
        indexb += 1;
      }
      if(!$.pkgInArray(data.package, ["docker"])) {
        if(data.preparation.msvc != null &&
           $.pkgInArray(data.package, ["source", "cppsdk"])) {
          ret += "<div class=\"simple\">";
          ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Microsoft Visual Studio</h3>";
          ret += "<p>" + data.preparation.msvc + "</p>";
          ret += "</div>";
          indexb += 1;
        }
        if(data.preparation.msvc_redist != null &&
           !$.pkgInArray(data.package, ["source", "cppsdk"])) {
          ret += "<div class=\"simple\">";
          ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Microsoft Visual C++ Redistributable</h3>";
          ret += "<p>" + data.preparation.msvc_redist + "</p>";
          ret += "</div>";
          indexb += 1;
        }
        if(data.preparation.basekit != null) {
          var applyto = data.system_requirements.software.basekit.applyto;
          if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
            ret += "<div class=\"simple\">";
            ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Intel® oneAPI Base Toolkit</h3>";
            var notes = [];
            // if(data.preparation.basekit.notinstall != null) {
            //   var note = "<p>Do NOT install the following Intel® oneAPI Base Toolkit components:</p>";
            //   note += $.ul_gen(data.preparation.basekit.notinstall);
            //   notes.push(note);
            // }
            var activate_script = "setvars.sh";
            if(data.os == "Windows")
              activate_script = "setvars.bat";
            var note = "<p>Use either individual component-specific activation scripts to activate required components listed below one-by-one, or use the oneAPI bundle script <code>" + activate_script + "</code> to activate the whole oneAPI environment. Check the <i>Sanity Test</i> section below for a usage example.</p>";
            if($.pkgInArray(data.package, ["source"])) {
              var note = "<p>Use individual component-specific activation scripts to activate required components listed below one-by-one. Check the <i>Sanity Test</i> section below for a usage example.</p>";
            }
            notes.push(note);
            if(data.preparation.msvc != null &&
               $.pkgInArray(data.package, ["source", "cppsdk"]))
              notes.push("Make sure the installation includes Miscrosoft C++ Build Tools integration.");
            if(data.preparation.basekit.notes != null)
              notes = $.merge(notes, data.basekit.notes);
            ret += $.notes_gen(notes);

            if(data.preparation.basekit.install != null) {
              ret += "<p>The following Intel® oneAPI Base Toolkit components are required:</p>";
              var components = [];
              $.each(data.preparation.basekit.install, function(index, value) {
                placeholder_example = "";
                if(value.placeholder_example != null)
                  placeholder_example = ", <i>e.g. " + value.placeholder_example + "</i>";
                components.push(value.name + " (<b>Placeholder <cite>" + value.placeholder + "</cite></b> as its installation path" + placeholder_example + ")");
              });
              ret += $.ul_gen(components);

              if(data.os == "Linux/WSL2") {
                var linux_dist = [];
                $.each(data.preparation.basekit.install, function(index0, value0) {
                  if(value0.package != null && value0.package[data.os] != null) {
                    $.each(Object.keys(value0.package[data.os]), function(index1, value1) {
                      if(!$.pkgInArray(value1, linux_dist)) {
                        linux_dist.push(value1);
                      }
                    });
                  }
                });
                if(linux_dist.length > 0) {
                  ret += "<p>Recommend using a <b>package manager</b> like apt, yum or dnf to install the packages above. <b>Follow instructions at <a href=\"https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux\">Intel® oneAPI Base Toolkit Download page</a> to setup the package manager repository.</b></p>";
                  ret += "<p>Take reference to commands below to install the packages on individual Linux distributions.</p>";
                  $.each(linux_dist, function(index0, value0) {
                    var packages = [];
                    $.each(data.preparation.basekit.install, function(index1, value1) {
                      if(value1.package != null && value1.package[data.os] != null && value1.package[data.os][value0] != null) {
                        packages.push(value1.package[data.os][value0]);
                      }
                    });
                    if(packages.length > 0) {
                      ret += "<p>* " + value0 + "</p>";
                      var command = "";
                      if(value0 == "Ubuntu")
                        command += "sudo apt install -y ";
                      if(value0 == "RHEL")
                        command += "sudo dnf install -y ";
                      if(value0 == "SUSE")
                        command += "sudo zypper install -y --oldpackage ";
                      command += packages.join(" ");
                      ret += $.code_gen([command]);
                    }
                  });
                }
              }
            }
            ret += "</div>";
            indexb += 1;
          }
        }
        if(data.system_requirements.software.basekit_hotfix != null) {
          var applyto = data.system_requirements.software.basekit.applyto;
          if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
            ret += "<div class=\"simple\">";
            ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Intel® oneAPI Base Toolkit Hotfix</h3>";
            ret += "<p>Follow the link above to install Intel® oneAPI Base Toolkit Hotfix</p>";
            ret += "</div>";
            indexb += 1;
          }
        }
        if(data.system_requirements.software.pti != null) {
          var applyto = data.system_requirements.software.basekit.applyto;
          if(applyto == null || applyto != null && $.pkgInArray(data.package, applyto)) {
            ret += "<div class=\"simple\">";
            ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Install Intel® Profiling Tools Interfaces for GPU (PTI for GPU)</h3>";
            ret += "<p>Follow the link above to download and install Intel® Profiling Tools Interfaces for GPU (PTI for GPU).</p>";
            ret += "<p>The following Intel® Profiling Tools Interfaces for GPU (PTI for GPU) component is required:</p>";
            var components = [];
            $.each(data.preparation.pti.install, function(index, value) {
              placeholder_example = "";
              if(value.placeholder_example != null)
                placeholder_example = ", <i>e.g. " + value.placeholder_example + "</i>";
              components.push(value.name + " (<b>Placeholder <cite>" + value.placeholder + "</cite></b> as its installation path" + placeholder_example + ")");
            });
            ret += $.ul_gen(components);
            ret += "</div>";
            indexb += 1;
          }
        }
      }

      ret += "</div>";
      indexa += 1;
    }
    if(data.installation != null) {
      ret += "<div class=\"simple\">";
      ret += "<h2>" + $.secid_gen([indexa]) + "Installation</h2>";
      if(data.version_mapping != null) {
        ret += $.notes_gen(["Intel® Extension for PyTorch* " + data.version_mapping.ipex + " has to work with PyTorch*/libtorch " + data.version_mapping.pytorch + "."]);
      }

      if($.pkgInArray(data.package, ["pip", "conda"])) {
        if(data.installation.notes != null)
          ret += $.notes_gen(data.installation.notes);
        var indexb = 0;
        if(data.installation.repos != null) {
          if(data.installation.repos.length > 1) {
            ret += "<p>Prebuilt wheel files are provided via multiple channels. You can choose either to install the packages.</p>";
          }
          $.each(data.installation.repos, function(index, value) {
            ret += "<div class=\"simple\">";
            if(value.repo != null) {
              var repo = value.repo;
              if(value.url != null)
                repo = "<a class=\"reference external\" href=\"" + value.url + "\">" + repo + "</a>";
              ret += "<h3>" + $.secid_gen([indexa], indexb) + repo + "</h3>";
            }
            ret += "<p>Prebuilt wheel files are available for Python " + value.python.join(", ") + ".";
            if(value.commands.length == 1) {
              ret += $.code_gen(value.commands[0]);
            } else {
              ret += $.notes_gen(["Should you experience low downloading speed issue, try choose another storage below."]);
              ret += "<div class=\"row\" id=\"row-storage" + index + "\" style=\"width: 50%\">";
              var block_n = value.commands.length;
              $.each(value.commands, function(index, value) {
                var selected = "";
                if(index == 0)
                  selected = " selected";
                ret += "<div class=\"values-element elem-instruction" + selected + " block-" + block_n + "\">Storage " + index + "</div>";
              });
              ret += "</div>";
              ret += "<p></p>";
              ret += "<div id=\"code-storage" + index + "\">";
              $.each(value.commands, function(index, value) {
                if(index == 0)
                  ret += $.code_gen(value).replace("code-div\"", "code-div\" style=\"display: flex\"");
                else
                  ret += $.code_gen(value).replace("code-div\"", "code-div\" style=\"display: none\"");
              });
              ret += "</div>";
            }
            if(value.notes != null)
              ret += $.notes_gen(value.notes);
            ret += "</div>";
            indexb += 1;
          });
        }
      } else if(data.package == "source") {
        if(data.installation.notes != null)
          ret += $.notes_gen(data.installation.notes);
        if(data.installation.aot != null)
          ret += $.notes_gen(["Refer to <a class=\"reference external\" href=\"" + data.installation.aot + "\">AOT documentation</a> for how to configure <cite>USE_AOT_DEVLIST</cite>. Without configuring AOT, the start-up time for processes using Intel® Extension for PyTorch* will be long, so this step is important."]);
        ret += "<p>To ensure a smooth compilation, a script is provided. If you would like to compile the binaries from source, it is highly recommended to utilize this script.</p>";
        if(data.installation.commands != null)
          ret += $.code_gen(data.installation.commands);
        ret += "<p>An example usage flow of this script is shown as below. After a successful compilation, you should see similar directory structure.</p>";
        if(data.installation.outputs != null)
          ret += $.code_gen(data.installation.outputs);
      } else if(data.package == "docker") {
        var indexb = 1;
        var indexc = -1;
        if(data.installation.build != null && data.installation.images != null)
          indexc = 0;
        if(data.installation.build != null) {
          var secid = "";
          if(indexc == -1)
            secid = $.secid_gen([indexa, indexb]);
          else {
            secid = $.secid_gen([indexa, indexb], indexc);
            indexc += 1;
          }
          ret += "<h3>" + secid + "Build Docker Image</h3>";
          $.each(Object.keys(data.installation.build), function(index, key) {
            if(key == "prebuilt") {
              ret += "<h4>* Install from prebuilt wheel files</h4>";
            }
            if(key == "compile") {
              ret += "<h4>* Compile from source</h4>";
            }
            value = data.installation.build[key];
            if(value.desc != null)
              ret += "<p>" + value.desc + "</p>";
            if(value.commands != null)
              ret += $.code_gen(value.commands);
          });
        }
        if(data.installation.images != null) {
          var secid = "";
          if(indexc == -1)
            secid = $.secid_gen([indexa, indexb]);
          else {
            secid = $.secid_gen([indexa, indexb], indexc);
            indexc += 1;
          }
          ret += "<h3>" + secid + "Pull Docker Image</h3>";
          $.each(Object.keys(data.installation.images), function(index, key) {
            ret += "<h4>* " + key + "</h4>";
            value = data.installation.images[key];
            if(value.desc != null)
              ret += "<p>" + value.desc + "</p>";
            if(value.commands != null)
              ret += $.code_gen(value.commands);
          });
        }
        indexb += 1;
        if(data.installation.run != null) {
          ret += "<h3>" + $.secid_gen([indexa, indexb]) + "Run Docker Image</h3>";
          if(data.installation.run.desc != null)
            ret += "<p>" + data.installation.run.desc + "</p>";
          ret += $.code_gen(data.installation.run.commands);
        }
      } else if(data.package == "cppsdk") {
        if(data.installation.files != null) {
          if(data.installation.files.length == 1) {
            var headers = [];
            var values = [];
            $.each(data.installation.files[0], function(index, value) {
              headers.push(value.abi);
              var vs = [];
              $.each(value.files, function(index, fn) {
                vs.push("<a href=\"" + fn.url + "\">" + fn.filename + "</a>");
              });
              values.push(vs.join(",&nbsp;"));
            });
            ret += $.table_1d_gen(headers, values);
          } else {
            ret += $.notes_gen(["Should you experience low downloading speed issue, try choose another storage below."]);
            ret += "<div class=\"row\" id=\"row-cppsdk\" style=\"width: 50%\">";
            var block_n = data.installation.files.length;
            $.each(data.installation.files, function(index, value) {
              var selected = "";
              if(index == 0)
                selected = " selected";
              ret += "<div class=\"values-element elem-instruction" + selected + " block-" + block_n + "\">Storage " + index + "</div>";
            });
            ret += "</div>";
            ret += "<p></p>";
            ret += "<div id=\"code-cppsdk\">";
            $.each(data.installation.files, function(index, value) {
              var table_cppsdk = "";
              var headers = [];
              var values = [];
              $.each(value, function(index, v) {
                headers.push(v.abi);
                var vs = [];
                $.each(v.files, function(index, fn) {
                  vs.push("<a href=\"" + fn.url + "\">" + fn.filename + "</a>");
                });
                values.push(vs.join(",&nbsp;"));
              });
              table_cppsdk = $.table_1d_gen(headers, values);
              if(index == 0)
                ret += table_cppsdk.replace("docutils\"", "docutils\" style=\"display: block\"");
              else
                ret += table_cppsdk.replace("docutils\"", "docutils\" style=\"display: none\"");
            });
            ret += "</div>";
          }
          if(data.installation.commands != null) {
            $.each(data.installation.commands, function(index, value) {
              var cmds = [];
              $.each(value.commands, function(index, v) {
                cmds.push(v);
              });
              ret += $.notes_gen([value.usage + $.code_gen(cmds)], "Usage");
            });
          }
        }
      } else {
        // Do nothing
      }

      ret += "</div>";
      indexa += 1;
    }
    if(data.sanity_test != null &&
       !$.pkgInArray(data.package, ["cppsdk"])) {
      ret += "<div class=\"simple\">";
      ret += "<h2>" + $.secid_gen([indexa]) + "Sanity Test</h2>";
      ret += "<p>You can run a simple sanity test to double confirm if the correct version is installed, and if the software stack can get correct hardware information onboard your system. The command should return PyTorch* and Intel® Extension for PyTorch* versions installed, as well as GPU card(s) information detected.</p>";
      var commands = [];
      var print_flag = false;
      if(data.system_requirements.software.basekit != null &&
         data.preparation.basekit.install != null &&
         !$.pkgInArray(data.package, ["docker"]) &&
         data.system_requirements.software.basekit.applyto == null) {
        print_flag = true;
      }
      if(print_flag) {
        ret += "<p>Check section <b><i>Install Intel® oneAPI Base Toolkit</i></b> ";
        if(data.preparation.pti != null) {
          ret += "and <b><i>Install Intel® Profiling Tools Interfaces for GPU (PTI for GPU)</i></b> ";
        }
        ret += "for <b>placeholders</b> used below.</p>";
        $.each(data.preparation.basekit.install, function(index, value) {
          commands.push(value.env);
        });
        if(data.preparation.pti != null) {
          $.each(data.preparation.pti.install, function(index, value) {
            commands.push(value.env);
          });
        }
      }
      commands.push(data.sanity_test);
      ret += $.code_gen(commands);
      m = data.version.match("v([0-9\\.]+\\+([a-zA-Z]+))");
      if(m != null && m.length == 3) {
        ret += "<p>Once installation succeeds, visit <a href=\"./" + m[2] + "/" + m[1] + "/tutorials/getting_started.html\">Get Started</a> and <a href=\"./" + m[2] + "/" + m[1] + "/tutorials/examples.html\">Examples</a> sections to start using the extension in your code.</p>";
      }
      ret += "</div>";
      indexa += 1;
    } else {
      // Do nothing
    }
    return ret;
  }

  $.row_append = function(items, num_elem = 0) {
    var stage = items[0];
    var search_id = -2;
    var search_val = "";
    if(search_fields[stage.toLowerCase()] != null)
      search_id = -1;
    items.shift();
    var row = document.createElement("div");
    row.className = "row";
    row.setAttribute("id", "row-" + stage.toLowerCase());
    var mh = document.createElement("div");
    mh.className = "mobile-headings";
    mh.innerHTML = stage;
    row.append(mh);

    if(num_elem == 0) {
      $.each(items, function(index, value) {
        if(!value.startsWith("comment-"))
          num_elem += 1;
      });
      idx = 0;
      $.each(items, function(index, value) {
        if(!value.startsWith("comment-")) {
          var elem = document.createElement("div");
          elem.className = "values-element block-" + num_elem + " install-" + stage.toLowerCase();
          elem.setAttribute("id", stage.toLowerCase() + "-" + value.toLowerCase());
          elem.innerHTML = value;
          row.append(elem);
          idx += 1;
          if(search_fields[stage.toLowerCase()] == value.toLowerCase())
            search_id = idx;
        }
      });
    } else {
      var elem_sel = document.createElement("select");
      elem_sel.className = "values-element block-" + num_elem + " install-" + stage.toLowerCase();
      elem_sel.setAttribute("id", stage.toLowerCase() + "-options");
      var opt = document.createElement("option");
      opt.setAttribute("value", "na");
      txt_history = "";
      if(stage.toLowerCase() == "version")
        txt_history = "history ";
      opt.innerHTML = "select a " + txt_history + stage.toLowerCase();
      elem_sel.append(opt);
      $.each(items, function(index, value) {
        if(index < num_elem - 1) {
          var elem = document.createElement("div");
          elem.className = "values-element block-" + num_elem + " install-" + stage.toLowerCase();
          elem.setAttribute("id", stage.toLowerCase() + "-" + value.toLowerCase());
          elem.innerHTML = value;
          row.append(elem);
          if(search_fields[stage.toLowerCase()] == value.toLowerCase())
            search_id = index + 1;
        } else {
          var opt = document.createElement("option");
          opt.setAttribute("value", value);
          opt.innerHTML = value;
          elem_sel.append(opt);
          if(search_fields[stage.toLowerCase()] == value.toLowerCase())
            search_val = value;
        }
      });
      row.append(elem_sel);
    }
    $("#col-headings").append("<div class=\"headings-element\">" + stage + "</div>");
    $("#col-values").append(row);

    if(search_id == -2) {
      if(num_elem == 1) {
        $("#col-values").children().last().children()[1].click();
      }
    } else {
      if(search_id == -1 && search_val == "") {
        alert(search_fields[stage.toLowerCase()] + " is not available for " + stage.toLowerCase());
        search_fields = [];
      } else if(search_id != -1) {
        $("#col-values").children().last().children()[search_id].click();
        delete search_fields[stage.toLowerCase()];
      } else {
        $("#" + stage.toLowerCase() + "-options").val(search_val).trigger("change");
        delete search_fields[stage.toLowerCase()];
      }
    }
  }

  $.query_gen = function(elem) {
    var query = {};
    elem.parent().parent().children().each(function() {
      var elem = $(this).find(".selected");
      var id = elem.attr("id").split("-");
      var category = id[0];
      var value = id[1];
      if(value == "options")
        value = elem.val();
      else
        value = elem.html();
      query[category] = value;
    });
    return query;
  }

  $.ajax_query = function(query) {
    query_json = JSON.parse(query);
    var hashes = [];
    for(var key in query_json) {
        if(query_json.hasOwnProperty(key)) {
            var pair = encodeURIComponent(key) + "=" + encodeURIComponent(query_json[key]);
            hashes.push(pair.toLowerCase());
        }
    }
    window.location.hash = "#installation?" + hashes.join("&");
    var formdata = new FormData();
    formdata.append("query", query);
    $.ajax({
      cache: false,
      type: "POST",
      url: "https://pytorch-extension.intel.com/installation-guide",
      data: formdata,
      processData: false,
      contentType: false,
      error: function(jqXHR, textStatus, errorThrown) {
        console.log("error");
        console.log(textStatus);
        console.log(errorThrown);
      },
      success: function(data) {
        var ret = jQuery.parseJSON(data);
        if(ret.stage == "Instruction") {
          $("#install-instructions").html($.instruction_gen(ret["data"]));
        } else {
          var num_elem = 0;
          if(ret.stage == "Version") {
            num_elem = 3;
            if(ret["data"][0].indexOf("Main") == -1)
              num_elem -= 1;
          }
          $.row_append($.merge([ret.stage], ret.data), num_elem);
        }
      }
    });
  }

  $.reset_selection = function(elem) {
    elem.parent().nextAll().remove();
    $("#col-headings").children().eq(elem.parent().index()).nextAll().remove();
    $("#install-instructions").html("");

    elem.parent().children().each(function() {
      $(this).removeClass("selected");
    });
  }

  $("#col-values").on("click", ".values-element", function() {
    fields = $(this).attr("id").split("-");
    if(fields[1] == "options")
      return;
    $("#" + fields[0] + "-options").val("na").trigger("change");
    $.reset_selection($(this));
    $(this).addClass("selected");

    var query = $.query_gen($(this))
    $.ajax_query(JSON.stringify(query));
  });

  $("#col-values").on("change", "select", function() {
    $.reset_selection($(this));
    if($(this).val() != "na") {
      $(this).addClass("selected");

      var query = $.query_gen($(this))
      $.ajax_query(JSON.stringify(query));
    }
  });

  $("#install-instructions").on("click", ".elem-instruction", function() {
    $(this).parent().children().each(function() {
      $(this).removeClass("selected");
    });
    $(this).addClass("selected");

    var index = $(this).index();
    var category = $(this).parent().attr("id").split("-")[1];
    var i = 0;
    $("#code-" + category).children().each(function () {
      if(i == index)
        $(this).css("display", "flex");
      else
        $(this).css("display", "none");
      i += 1;
    });
  });

  $("#install-instructions").on("click", ".svg-copy", function() {
    try {
      var text = $(this).parent().parent().children("div:first").children("div:first").text();
      navigator.clipboard.writeText(text);
      var div_svg = $(this).parent();
      div_svg.html(svg_done);
      setTimeout(function(div) {
        div.html(svg_copy);
      }, 2000, div_svg);
    } catch(err) {
      window.alert("Failed to copy commands to clipboard.\n" + err.message);
    }
  });
});
