$(document).ready(function() {
  var hash_list = "";
  $("#menu-introduction").addClass("selected");
  if(window.location.hash.startsWith("#installation")) {
    hash_list = decodeURIComponent(window.location.hash).split("?");
    setTimeout(function () {
      $("#menu-installation").trigger("click");
    }, 10);
  // } else {
  //   setTimeout(function () {
  //     $("#menu-introduction").trigger("click");
  //   }, 10);
  }

  $(".menu-element").on("click", function() {
    $(this).parent().children().each(function() {
      $(this).removeClass("selected");
    });
    $(this).addClass("selected");

    if($(this).attr("id") == "menu-installation") {
      // $("#div-introduction").hide();
      // $("#div-installation").show();
      // if(!window.location.hash.startsWith("#installation"))
      //   window.location.hash = "#installation";
      let href = "https://pytorch-extension.intel.com/installation";
      if(hash_list.length > 1)
        href += "?" + hash_list[1];
      // window.open(href, '_blank');
      // window.location.hash = "#introduction";
      window.location.href = href;
      $("#menu-introduction").addClass("selected");
      $("#menu-installation").removeClass("selected");
    } else if($(this).attr("id") == "menu-introduction") {
      $("#div-introduction").show();
      $("#div-installation").hide();
      if(!window.location.hash.startsWith("#introduction"))
        window.location.hash = "#introduction";
    } else {
      // Do nothing
    }
  });
});
