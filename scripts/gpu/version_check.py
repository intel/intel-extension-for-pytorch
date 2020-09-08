import sys
import os
import re
import difflib
import torch

class VersionAnalyzer():
    def __init__(self):
        self.Pattern = ' (\w+)\('
        self.OpModifications = {}
        self.DiffOp = []
        self.AddedOp = []
        self.RemovedOp = []
        
        try:
            torch_path = os.path.dirname(torch.__file__)
            self.THAtenRegistDeclare = os.path.join(torch_path,"include","torch","csrc","autograd","generated","RegistrationDeclarations.h")
            cwd = os.path.dirname(os.path.abspath(__file__))
            self.ExtAtenRegistDeclare = os.path.join(cwd, "reference", "RegistrationDeclarations.h")
        except Exception:
            sys.exit('Cannot find operation registration declaration!')

    def analysis(self):

        diff = difflib.unified_diff(open(self.THAtenRegistDeclare).readlines(), open(self.ExtAtenRegistDeclare).readlines())
        result = '\n'.join(x.strip() for x in diff if not x.startswith(' '))
        for line in result.splitlines():
            match = re.search(self.Pattern, line)
            if match and match.group(1) in self.OpModifications:
                if line.startswith('+'):
                    self.OpModifications[match.group(1)] += 1
                else:
                    self.OpModifications[match.group(1)] -= 1
            elif match and match.group(1) not in self.OpModifications:
                if line.startswith('+'):
                    self.OpModifications[match.group(1)] = 1
                else:
                    self.OpModifications[match.group(1)] = -1
    def report(self):
        if self.OpModifications is not {}:
            for key in self.OpModifications:
                self.DiffOp.append(key)
                if self.OpModifications[key] > 0 :
                    self.AddedOp.append(key)
                if self.OpModifications[key] < 0 :
                    self.RemovedOp.append(key)
        return self.DiffOp, self.AddedOp, self.RemovedOp

def main():
    analyzer = VersionAnalyzer()
    analyzer.analysis()
    op_list, new_op, removed_op = analyzer.report()
    print("Checking versions..")
    if len(op_list) > 0:
        print("Warning: The Pytorch Extension has conflicting operation declarations with currently installed Pytorch, you should consider upgrading them.")
        print("Warning: The following Op have conflicting definition: ", ', '.join(op_list))
    if len(new_op) > 0:
        print("Warning:", ', '.join(new_op), " is not supported in Pytorch!")
    if len(removed_op) > 0:
        print("Warning:", ', '.join(removed_op), " is not supported in Pytorch Extension!")
    print("Finished checking!")

if __name__ == '__main__':
    main()