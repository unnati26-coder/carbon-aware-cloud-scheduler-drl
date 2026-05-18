import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class WorkloadGenerator {
    private static final int NUM_HOSTS = 20;
    private static final int NUM_TASKS = 1000;
    private static final Random rand = new Random(42);

    public static void main(String[] args) throws Exception {
        CloudSimPlus simulation = new CloudSimPlus();

        // Create hosts (servers)
        List<Host> hosts = createHosts();
        DatacenterSimple datacenter = new DatacenterSimple(simulation, hosts);

        // Create broker and VMs
        DatacenterBrokerSimple broker = new DatacenterBrokerSimple(simulation);
        List<Vm> vms = createVms(NUM_TASKS);  // 1 VM per task guarantees all run
        List<Cloudlet> cloudlets = createCloudlets(NUM_TASKS);

        broker.submitVmList(vms);
        broker.submitCloudletList(cloudlets);

        simulation.start();

        // Export results to CSV for Python
        exportToCsv(broker.getCloudletFinishedList(), "tasks.csv");
        System.out.println("Exported " + NUM_TASKS + " tasks to tasks.csv");
        System.exit(0);
    }

    static List<Host> createHosts() {
        List<Host> list = new ArrayList<>();
        int[] cpuTypes = {2, 4, 8, 16}; // micro, small, medium, large
        long[] ramTypes = {2048, 4096, 8192, 16384};

        for (int i = 0; i < NUM_HOSTS; i++) {
            int type = i % 4;
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < cpuTypes[type]; j++) {
                peList.add(new PeSimple(1000));
            }
            Host host = new HostSimple(ramTypes[type], 100000, 100000, peList);
            list.add(host);
        }
        return list;
    }

    static List<Vm> createVms(int count) {
        List<Vm> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            list.add(new VmSimple(1000, 1 + rand.nextInt(4))
                .setRam(512 + rand.nextInt(4096))
                .setBw(1000)
                .setSize(10000));
        }
        return list;
    }

    static List<Cloudlet> createCloudlets(int count) {
        List<Cloudlet> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            long length = 1000 + rand.nextInt(9000);    // MI (task size)
            int cpu = 1 + rand.nextInt(4);               // CPU cores
            long ram = 128 + rand.nextInt(1024);         // MB

            Cloudlet c = new CloudletSimple(length, cpu);
            c.setFileSize(300).setOutputSize(300);
            list.add(c);
        }
        return list;
    }

    static void exportToCsv(List<Cloudlet> cloudlets, String filename) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(filename));
        pw.println("task_id,cpu_demand,memory_demand,length,arrival_time," +
                   "start_time,finish_time,deadline,status");

        int i = 0;
        for (Cloudlet c : cloudlets) {
            double arrival  = c.getSubmissionDelay();
            double start    = c.getExecStartTime();
            double finish   = c.getFinishTime();
            double deadline = finish + rand.nextDouble() * 10; // SLA window
            int    cpu      = (int) c.getPesNumber();
            long   ram      = 128 + rand.nextInt(1024);

            pw.printf("%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%s%n",
                i++, cpu, ram, c.getLength(),
                arrival, start, finish, deadline,
                c.getStatus().name());
        }
        pw.close();
    }
}