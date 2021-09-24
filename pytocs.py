#send instrustions array in form of C#
def updatePath(instructions):
    file_ = open("firmware.ino", "r")
    message = ""
    for row in file_:
        if ("loop()" not in row):
            #print(row)
            message = message + row
        else:
            break
    file_.close()

    file_ = open("firmware.ino", "w")
    message = message + "void loop(){"

    #instructions
    instructions_code = ""
    for instruction in instructions:
        if instruction == "left":
            instructions_code = instructions_code + " NinetyLeft();"
        if instruction == "right":
            instructions_code = instructions_code + " NinetyRight();"        
        if instruction == "forward":
            instructions_code = instructions_code + " OneSpaceForward();"

    #instructions end

    message  = message + instructions_code + " return;}"
    file_.write(message)
    file_.close()


