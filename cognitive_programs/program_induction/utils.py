import numpy as np

from cognitive_programs.data_generation.primitive_shapes import ALL_SHAPE_NAMES, \
    ALL_COLORS

START_NODE = 0


def create_commands_dict():

    commands_dict = {
        0: ("em.scene_parse", None),
        1: ("em.top_down_attend", None),
        5: ("em.set_shape_attn", ALL_SHAPE_NAMES),
        6: ("em.set_color_attn", ALL_COLORS),
        16: ("em.reset_attn", None),
        23: ("em.attend_pointer", None),
        3: ("em.grab_object", None),
        12: ("em.release_object", None),
        14: ("em.next_object", None),
        19: ("em.move_hand_to_fixate_collision", None),
        2: ("em.move_hand_to_attended_object", None),
        4: ("em.move_hand_to_pointer", None),
        7: ("em.move_hand_left", None),
        8: ("em.move_hand_right", None),
        9: ("em.move_hand_up", None),
        10: ("em.move_hand_down", None),
        13: ("em.fill_color", ALL_COLORS),
        11: ("em.fixate", ['left', 'right', 'bottom', 'top', 'center']),
        # 15: (em.get_color, None),
        # 24: (em.get_shape, None),
        17: ("em.fixate_attended_object", None),
        18: ("em.fixate_previous", None),
        20: ("em.fixate_next", None),
        21: ("em.fixate_pointer", None),
        22: ("em.remove_from_imagine_buffer", None),
        25: ("em.loop_start", None),
        26: ("em.loop_end", None),
        27: ("em.reset_imagination", None),
        28: ("em.imagine_object", ['circle_shape', 'hline_shape']),  # 'vline_shape'
        # 29: (em.set_shape_attn, ['internal']),
        # 30: (em.set_color_attn, ['internal']),
        # write_to_imagination
    }
    return commands_dict


def create_args(commands_dict, inst_list,
                att_shape_arg=None,
                att_color_arg=None,
                change_shape_arg=None,
                change_color_arg=None,
                fixate_arg=None):
    args = []
    for c in inst_list:
        arg = create_one_func_signature(c,
                                        commands_dict,
                                        att_shape_arg,
                                        att_color_arg,
                                        change_shape_arg,
                                        change_color_arg,
                                        fixate_arg)
        args.append(arg)
    return args


def create_one_func_signature(c, commands_dict,
                              att_shape_arg=None,
                              att_color_arg=None,
                              change_shape_arg=None,
                              change_color_arg=None,
                              fixate_arg=None):
    if commands_dict[c][0].__name__ == 'set_shape_attn':
        return att_shape_arg
    elif commands_dict[c][0].__name__ == 'set_color_attn':
        return att_color_arg
    elif commands_dict[c][0].__name__ == 'fill_color':
        return change_color_arg
    elif commands_dict[c][0].__name__ == 'change_shape':
        return change_shape_arg
    elif commands_dict[c][0].__name__ == 'fixate':
        return fixate_arg
    else:
        return None


def exec_program(em, commands_dict, inst_list, args_list,
                 robot_interface=None, subroutines={}, verbose=False):
    """ This function modifies em in-place, then redundantly returns the modified em"""
    # If the first instruction is not scene_parsing, add scene_parse as the
    # first instruction before executing other instructions
    for i, c in enumerate(inst_list):
        arg = args_list[i]
        is_success = exec_one_action(
            c, arg, commands_dict, robot_interface, subroutines, verbose)
        if not is_success:
            if verbose:
                print("Failed at {}:{}".format(c, commands_dict[c]))
            return (i, None)
    return (-1, em)


def exec_program_with_loops(em, commands_dict, inst_list, args_list,
                            robot_interface=None, subroutines={},
                            verbose=False):
    """ Note that if the input inst_list is not closed loop and without
    """
    if not check_loop_closure(inst_list):
        if verbose:
            print "invalid loop"
        return (-1, None)   # Don't skip, but no emulator.

    i = 0
    while i < len(inst_list):
        inst = inst_list[i]
        arg = args_list[i]

        # Execute it
        is_success = exec_one_action(inst, arg, commands_dict, robot_interface,
                                     subroutines,
                                     verbose=verbose)
        if not is_success:
            if verbose:
                print "----------Aborted at step {} ({} {})".format(i, inst,
                                                                    arg)
            return (i, None)  # failed-instruction, emulator

        # Check if instruction is the start of a loop
        if inst != 25:  # loop start
            i += 1
        else:
            # this is the start of a loop.
            # Scan forward to find the end.
            loop_end = len(inst_list)
            for k in range(i + 1, len(inst_list)):
                if inst_list[k] == 26:
                    # That is the end of the loop!
                    loop_end = k
                    break
            # At this point we have the end of the loop.
            # Now we execute the loop

            loop_block = inst_list[i + 1: loop_end]  # Exclude the actual end
            args_block = args_list[i + 1: loop_end]
            if verbose:
                print "loop block", enumerate_program(commands_dict, loop_block)

            loop_valid, loop_fail_idx = check_loop_validity(loop_block)

            if not loop_valid:
                return (i + loop_fail_idx + 1, None)

            # Now execute the loop:
            if verbose:
                print "Starting loop ........{"

            max_num_loop = len(em.attended_obj_indices) + 1
            loop_count = 0

            while em.att_obj_idx < len(em.attended_obj_indices):
                failed_inst, new_em = exec_program(em, commands_dict, loop_block,
                                                   args_block,
                                                   robot_interface,
                                                   subroutines=subroutines,
                                                   verbose=verbose)
                if failed_inst >= 0:
                    if verbose:
                        print "-----Aborted"
                    return (i + failed_inst, None)
                loop_count += 1
                if loop_count > max_num_loop:
                    if verbose:
                        print "breaking out of infinite loop"
                    break
            if verbose:
                print " Ending loop ..........}"

            i = loop_end

    if verbose:
        print "------Finished"
    return (-1, em)   # No failed instruction, emulator


def check_loop_closure(inst_list):
    # for now this is a minimal implementation
    if inst_list.count(25) != inst_list.count(26):
        return False

    loop_start_indices = np.where(np.array(inst_list) == 25)[0]
    loop_end_indices = np.where(np.array(inst_list) == 26)[0]

    for k in range(len(loop_start_indices)):
        if loop_end_indices[k] <= loop_start_indices[k]:
            return False

    return True


def enumerate_program(commands_dict, inst_list, att_shape_arg=None,
                      att_color_arg=None,
                      change_shape_arg=None, change_color_arg=None,
                      fixate_arg=None):
    args_list = create_args(commands_dict,
                            inst_list,
                            att_shape_arg,
                            att_color_arg,
                            change_shape_arg,
                            change_color_arg,
                            fixate_arg)
    prog_list = []
    for i, c in enumerate(inst_list):
        arg = args_list[i]
        cmd = commands_dict[c][0]
        arg = args_list[i]
        prog_list.append(cmd.__name__ + " " + str(arg))
    return prog_list


def check_loop_validity(inst_list):
    """ Given a list of instructions, check whether they can form a valid loop.
    This means, checking for anything that could create an infinite loop.
    We are also disallowing double loops right now"""

    for i, c in enumerate(inst_list):
        if c in [5, 6, 16, 25]:
            return False, i
    return True, -1


def exec_one_action(c, arg, commands_dict, robot_interface=None,
                    subroutines={}, verbose=False):
    """ Execute an action, where it could be a subroutine or a primitive
    action. The arguments for a subroutine is ordered by a dfs traversal on
    the hierarchy"""
    if c in subroutines:
        c_children = subroutines[c]
        for c_child in c_children:
            if c_child in subroutines:
                is_success = exec_one_action(c_child, arg, commands_dict,
                                             robot_interface, subroutines,
                                             verbose)
            else:
                is_success = exec_one_primitive_action(c_child, arg[0],
                                                       commands_dict, robot_interface,
                                                       verbose)
                arg = arg[1:]

            if not is_success:
                return False
        return True
    else:
        return exec_one_primitive_action(c, arg, commands_dict, robot_interface, verbose)


def exec_one_primitive_action(c, arg, commands_dict, robot_interface=None,
                              verbose=False):
    if arg is None:
        try:
            commands_dict[c][0]()
        except Exception:
            return False
    else:
        try:
            commands_dict[c][0](*[arg])
        except Exception:
            return False
    if robot_interface is not None and c in robot_interface.known_codes:
        robot_interface.execute_by_code(c, arg)
    return True


def augment_all_programs_with_non_arg(programs_without_non_arg):
    programs_with_non_arg = {}
    for (task_name, program_without_non_arg) in programs_without_non_arg.iteritems():
        program_with_non_arg = augment_with_non_arg(program_without_non_arg)
        programs_with_non_arg[task_name] = program_with_non_arg
    return programs_with_non_arg


def augment_with_non_arg(program_without_non_arg):
    program_with_non_arg = []
    for function in program_without_non_arg:
        if type(function) is int:
            program_with_non_arg.append((function, None))
        elif type(function) is tuple:
            program_with_non_arg.append(function)
        else:
            raise ValueError("Invalid type of function")
    return program_with_non_arg


def is_excluded_task(task_name, program):
    for inst, arg in program:
        if inst == 28 and not (arg == 'circle_shape' or arg == 'hline_shape'):
            return True
    return False
