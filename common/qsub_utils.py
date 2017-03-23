# qsub_utils

import os
import stat
import re
import subprocess
import platform
import logging

PBS_CONFIGURATION = {}
logger = logging.getLogger('qsub_utils')


def setup_configuration(nodes=''):
    """
    Method to setup PBS configuration
    :param nodes: specify nodes with list of features. `-l nodes={nodes}`
    """
    global PBS_CONFIGURATION   
    if len(nodes) > 0:
        PBS_CONFIGURATION['nodes'] = nodes
        

def get_configuration_str(conf_dict):
    """
    Method to convert configuration dictionary to string
    """
    conf_str = "#PBS"    
    if 'nodes' in conf_dict:
        conf_str += " -l nodes=%s" % conf_dict['nodes']
    if 'name' in conf_dict:
        conf_str += " -N %s" % conf_dict['name']
    if 'cwd' in conf_dict:         
        conf_str += " -d %s" % conf_dict['cwd'] 
    if 'stdout' in conf_dict:
        conf_str += " -o %s" % conf_dict['stdout']
    if 'stderr' in conf_dict:        
        conf_str += " -e %s" % conf_dict['stderr']           
    return conf_str

    
def write_launch_file(cmd_str, conf_dict, env=''):
    """
    Method to write a PBS launch file for qsub

    :param cmd_str: command string, e.g "python -c \'import sys; sys.print\'"
    :param conf_dict: configuration dictionary, {'name': 'aName', 'cwd':'path/to/dir/', 'nodes': 'nodes_conf',...}
    :param env: environmanet string, e.g. "export PATH=$PATH:/path/to/bin"
    """
    filename = conf_dict['launch']
    with open(filename, 'w') as w:        
        w.write(get_configuration_str(conf_dict) + '\n\n')
        if len(env) > 0:
            w.write(env + '\n')
        w.write(cmd_str)
    os.chmod(filename, stat.S_IRWXG | stat.S_IRWXU | stat.S_IRWXO)


def submit_job(cmd, name, cwd, env=''):
    """
    Method to submit a job writing a launch file and using qsub
    `qsub job_{name}.launch`

    :param cmd: list of commands, e.g. ['python', '-c', '\"import sys; print sys.path\"']
    :param name: name of the job
    :param cwd: current working directory
    :param env: environmanet string, e.g. "export PATH=$PATH:/path/to/bin"
    """
    assert len(name) > 0, "Job name can not be empty"
    assert len(cmd) > 0, "Job command can not be empty" 
    assert len(cwd) > 0, "Job working directory can not be empty"
    
    if ' ' in name:
        name = name.replace(' ', '_')
        name = name.replace('(', '_')
        name = name.replace(')', '_')

    filename = os.path.join(cwd, '%s.launch' % name)
        
    job_conf = dict()
    job_conf['nodes'] = PBS_CONFIGURATION['nodes']
    job_conf['name'] = name
    if len(cwd) > 0: job_conf['cwd'] = cwd
    job_conf['launch'] = filename
        
    write_launch_file(' '.join(cmd), job_conf, env)
    program = ['qsub', '-V', '-I', '-x', filename]
    process = subprocess.Popen(program,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    line = process.stdout.readline()    
    m = re.search("\d+.\w+", line)
    assert m is not None, "Job id is not found in the first line of output: %s" % line
    job_id = m.group(0)
    job_conf['id'] = job_id
    return process, job_conf


def delete_job(job_id):
    if not job_is_running(job_id):
        logger.warn("Job '%s' is not running. Can not delete job" % job_id)
        return False
    _id = _get_id(job_id)
    program = ['qdel', _id]
    process = subprocess.Popen(program,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               close_fds=False if platform.system() == 'Windows' else True)
    returncode = process.wait()
    return returncode == 0


def _get_id(job_id):
    _id = job_id.split('.')[0]
    return _id


def get_stats(job_id):
    _id = _get_id(job_id)
    program = ['qstat', '-f', _id]
    process = subprocess.Popen(program,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               close_fds=False if platform.system() == 'Windows' else True)
    process.wait()
    out = process.stdout.read()
    out = out.split('\n')
    stats = {}
    if len(out) > 0:
        for line in out:
            kv = line.split(' = ')
            if len(kv) > 1:
                stats[kv[0].strip()] = kv[1].strip()
    return stats


def job_is_running(job_id):
    stats = get_stats(job_id)
    if len(stats) > 0:
        """
        the job states
            E -    Job is exiting after having run.
            H -    Job is held.
            Q -    job is queued, eligable to run or routed.
            R -    job is running.
            T -    job is being moved to new location.
            W -    job is waiting for its execution time (-a option) to be reached.
            S -    (Unicos only) job is suspend.
        """
        if stats['job_state'] in ['R', 'Q', 'H', 'T', 'W']:
            return True
    else:
        return False


def get_stdout(job_info):
    filename = job_info['stdout']
    if not os.path.exists(filename):
        logger.warn("Stdout filename %s' is not found" % filename)
        return None
    out = []
    with open(filename, 'r') as r:
        while True:
            line = r.readline()
            if len(line) == 0:
                break
            out.append(line)
    return out


def get_stderr(job_info):
    filename = job_info['stderr']
    if not os.path.exists(filename):
        logger.warn("Stdout filename %s' is not found" % filename)
        return None
    out = []
    with open(filename, 'r') as r:
        while True:
            line = r.readline()
            if len(line) == 0:
                break
        out.append(line)
    return out
