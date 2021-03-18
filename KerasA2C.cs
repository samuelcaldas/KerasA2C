using System;
using System.Collections.Generic;
using NumSharp;

using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

using System.Linq;
using System.Threading;
namespace Tensorflow
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Ola");
        }
    }
    //inspired by Tensorflow sample of Actor Critic Method in Cartpole Environment
    internal class A2C
    {
        int seed = 42;
        float gamma = .99f;
        int max_steps_per_episode = 10000;
        int num_inputs = 4;
        int num_actions = 2;
        int num_hidden = 128;
        GymEnvironment env;
        //TODO: Python debugger 1.1920928955078125e-07 == np.finfo(np.float32).eps.item()
        //NET float epsilon = 1.401298E-45F
        float eps = 1.1920928955078125e-07F;
        public A2C()
        {
            env = GymEnvironment.make("CartPole-v0");
            env.seed(seed);
        }

        //TODO: allow input of action space, input space.

        Tensors inputs;
        Tensors common;
        Tensors action;
        Tensors critic;
        Tensors outputs;
        Keras.Engine.Functional model;
        private OptimizerV2 optimizer;
        private ILossFunc huber_loss;
        private List<Tensor> action_probs_history;
        private List<Tensor> critic_value_history;
        private List<float> rewards_history;
        private float running_reward;
        private int episode_count;

        float[] state;
        private float episode_reward;

        internal void Run()
        {

            //build the model;
            var layers = new LayersApi();
            inputs = keras.Input(num_inputs, dtype: TF_DataType.TF_FLOAT);
            common = layers.Dense(num_hidden, activation: "relu").Apply(inputs);
            action = layers.Dense(num_actions, activation: "softmax").Apply(common);
            critic = keras.layers.Dense(1).Apply(common);
            outputs = new Tensors(action, critic);
            model = keras.Model(inputs, outputs, name: "a2c");

            //optimizer = keras.optimizers.Adam(learning_rate = 0.01)
            optimizer = keras.optimizers.Adam(learning_rate: 0.01f);
            //huber_loss = keras.losses.Huber()
            huber_loss = keras.losses.Huber();
            //action_probs_history = []
            action_probs_history = new List<Tensor>();
            //critic_value_history = []
            critic_value_history = new List<Tensor>();
            //rewards_history = []
            rewards_history = new List<float>();
            //running_reward = 0
            running_reward = 0f;
            //episode_count = 0
            episode_count = 0;

            train();
        }

        void train()
        {

            while (true)// run until solved
            {
                var tensorstate = env.reset();
                episode_reward = 0;
                using (var tape = tf.GradientTape())
                {
                    for (var timestep = 1; timestep < max_steps_per_episode; timestep++)
                    {
                        //env.render //TODO:
                        var stateAsTensor2 = tf.expand_dims(tensorstate, axis: 0);

                        //Predict action probabilities and estimated future rewards
                        //from environment state
                        //(action_probs, critic_value) = model(state)
                        var result = model.predict(stateAsTensor2);
                        var action_probs = result[0];
                        var critic_value = result[1];
                        var output = critic_value.ToArray<float>();
                        critic_value_history.Add(critic_value[0, 0]);

                        var propabilities = action_probs.ToArray<double>();

                        // use custom random choice instead because
                        // np.random.choice throws a not implemented exception
                        // np.random.choice(num_actions, probabilities: propabilities);
                        var action = RandomChoice.Choice(num_actions, propabilities);

                        var loginput = propabilities[action];
                        var probLog = tf.math.log(action_probs[0, action]);
                        action_probs_history.append(probLog);

                        //state, reward, done, _ = env.step(action);
                        var stepResult = env.step(action);
                        tensorstate = stepResult.state;
                        state = stepResult.state.ToArray<float>();
                        var reward = stepResult.reward;
                        rewards_history.Add(reward);
                        episode_reward += reward;

                        //rewards_history.append(reward)
                        //episode_reward += reward
                        //action_probs_history.append(tf.math.log(action_probs[0, action]))
                        if (stepResult.done)
                            break;
                    }

                    // # Update running reward to check condition for solving
                    running_reward = 0.05f * episode_reward + (1 - 0.05f) * running_reward;


                    //# Calculate expected value from rewards
                    //# - At each timestep what was the total reward received after that timestep
                    //# - Rewards in the past are discounted by multiplying them with gamma
                    //# - These are the labels for our critic   
                    var returns = new List<float>();
                    var discounted_sum = 0f;
                    for (var i = 0; i < rewards_history.Count; i++)
                    {
                        var r = rewards_history[i];
                        discounted_sum = r + gamma * discounted_sum;
                        returns.Insert(0, discounted_sum);
                    }

                    var npReturns = np.array(returns.ToArray());
                    npReturns = (npReturns - np.mean(npReturns)) / (np.std(npReturns) + eps);
                    var returnsASList = npReturns.ToArray<float>();


                    //history = zip(action_probs_history, critic_value_history, returns)
                    var actor_losses = new List<Tensor>();
                    var critic_losses = new List<Tensor>();
                    for (var i = 0; i < action_probs_history.Count; i++)
                    {
                        var log_prob = action_probs_history[i];
                        var value = critic_value_history[i];
                        var ret = returns[i];

                        // At this point in history, the critic estimated that we would get a
                        // total reward = `value` in the future. We took an action with log probability
                        // of `log_prob` and ended up recieving a total reward = `ret`.
                        // The actor must be updated so that it predicts an action that leads to
                        // high rewards (compared to critic's estimate) with high probability.
                        var diff = ret - value;
                        actor_losses.Add(diff);

                        // The critic must be updated so that it predicts a better estimate of
                        // the future rewards.
                        var retTensor = tf.convert_to_tensor(ret);
                        var loss = huber_loss.Call(tf.expand_dims(value, 0), tf.expand_dims(retTensor, 0));
                        critic_losses.Add(loss);
                    }


                    //var loss_value = sum(actor_losses) + sum(critic_losses); //broken
                    var actor_losses_sum = actor_losses.SelectMany(x => x.ToArray<float>()).Sum();
                    var critic_losses_sum = critic_losses.SelectMany(x => x.ToArray<float>()).Sum();
                    // Backpropagation

                    float loss_value = actor_losses_sum + critic_losses_sum;
                    Tensor loss_value_tensor = tf.convert_to_tensor(loss_value);
                    var grads = tape.gradient(loss_value_tensor, model.trainable_variables);

                    //optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    optimizer.apply_gradients(zip(grads, model.trainable_variables.Select(x => x as ResourceVariable)));
                    // var zipped = grads.Zip(model.trainable_variables.Cast<ResourceVariable>(),(a,b)=>(a,b)).ToList();
                    // optimizer.apply_gradients(zipped);

                    // Clear the loss and reward history
                    action_probs_history.Clear();
                    critic_value_history.Clear();
                    rewards_history.Clear();
                }
                episode_count += 1;
                if (episode_count % 10 == 0)
                {
                    Console.WriteLine($"running reward: {running_reward.ToString("N2")} at {episode_count}");
                }
                if (running_reward > 195) //# Condition to consider the task solved
                {
                    print($"Solved at episode {episode_count}!");
                    break;
                }
            }

            //Log details



        }


    }
    public static class RandomChoice
    {


        //from: https://stackoverflow.com/a/43345968/624988
        static readonly ThreadLocal<Random> _random = new ThreadLocal<Random>(() => new Random());
        static IEnumerable<T> Choice<T>(IList<T> sequence, int size, double[] distribution)
        {
            double sum = 0;
            // first change shape of your distribution probablity array
            // we need it to be cumulative, that is:
            // if you have [0.1, 0.2, 0.3, 0.4] 
            // we need     [0.1, 0.3, 0.6, 1  ] instead
            var cumulative = distribution.Select(c =>
            {
                var result = c + sum;
                sum += c;
                return result;
            }).ToList();
            for (int i = 0; i < size; i++)
            {
                // now generate random double. It will always be in range from 0 to 1
                var r = _random.Value.NextDouble();
                // now find first index in our cumulative array that is greater or equal generated random value
                var idx = cumulative.BinarySearch(r);
                // if exact match is not found, List.BinarySearch will return index of the first items greater than passed value, but in specific form (negative)
                // we need to apply ~ to this negative value to get real index
                if (idx < 0)
                    idx = ~idx;
                if (idx > cumulative.Count - 1)
                    idx = cumulative.Count - 1; // very rare case when probabilities do not sum to 1 becuase of double precision issues (so sum is 0.999943 and so on)
                                                // return item at given index
                yield return sequence[idx];
            }
        }
        public static T Choice<T>(IList<T> sequence, double[] distribution)
        {
            return Choice(sequence, 1, distribution).First();
        }

        public static int Choice(int upTo, double[] distribution)
        {
            return Choice(Enumerable.Range(0, upTo).ToArray(), distribution);
        }
    }
class GymEnvironments
    {
        public const string CartPolev0 = "CartPole-v0";
    }

    public interface ILogger
    {
        void warn(string message);
    }
    public class ConsoleLogger : ILogger
    {
        public void warn(string message)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(message);
            Console.ResetColor();
        }
    }
    public abstract class GymEnvironment
    {
        protected ILogger logger = new ConsoleLogger();
        internal static GymEnvironment make(string v)
        {
            switch (v)
            {
                case GymEnvironments.CartPolev0:
                    return new CartPolev0();
                default:
                    throw new NotImplementedException();
            }
        }

        public abstract void seed(int seed);
        public abstract Tensor reset();


        public virtual EnvorinmentStepResult step(int action)
        {
            throw new NotImplementedException();
        }
    }

    public class CartPolev0 : GymEnvironment
    {
        private float gravity;
        private float masscart;
        private float masspole;
        private float total_mass;
        private float length;
        private float polemass_length;
        private float force_mag;
        private float tau;
        private string kinematics_integrator;
        private float theta_threshold_radians;
        private float x_threshold;
        private NDArray high;
        private spaces action_space;
        private spaces observation_space;
        private NDArray state;
        private int? steps_beyond_done;
        NumPyRandom rng;
        public CartPolev0()
        {
            this.gravity = 9.8f;
            this.masscart = 1.0f;
            this.masspole = 0.1f;
            this.total_mass = (this.masspole + this.masscart);
            this.length = 0.5f;  // actually half the pole's length
            this.polemass_length = (this.masspole * this.length);
            this.force_mag = 10.0f;
            this.tau = 0.02f;  //seconds between state updates
            this.kinematics_integrator = "euler";

            //# Angle at which to fail the episode
            this.theta_threshold_radians = (float)(12 * 2 * Math.PI / 360);
            this.x_threshold = 2.4f;

            // Angle limit set to 2 * theta_threshold_radians so failing observation
            // is still within bounds.
            var highValues = new float[] {(float)this.x_threshold * 2,
                         float.MaxValue, //np.finfo(np.float32).max,
                         (float)this.theta_threshold_radians * 2,
                         float.MaxValue}; //, //np.finfo(np.float32).max],
            this.high = np.array(highValues);

            var negHigh = high.negative();

            this.action_space = spaces.Discrete(2);
            this.observation_space = spaces.Box(negHigh, high, np.float32);

            //this.seed();
            //this.viewer = None;
            this.state = null;

            this.steps_beyond_done = null;
        }
        int _seed;
        public override void seed(int seed)
        {
            this._seed = seed;
            rng = np.random.RandomState(seed);
        }

        public override Tensor reset()
        {
            // random_ops.random_uniform(new int[] { }, minval: -0.05f, maxval: 0.05f);
            var result = rng.uniform(-0.05f, 0.05f, (4));
            var asFloat = result.astype(NPTypeCode.Float);
            this.state = asFloat; 
            steps_beyond_done = null;
            return np.array(asFloat);
        }


        public override EnvorinmentStepResult step(int action)
        {
            //    err_msg = "%r (%s) invalid" % (action, type(action))
            //assert self.action_space.contains(action), err_msg

            var stateAsArray = this.state.ToArray<float>();
            float x = stateAsArray[0];
            float x_dot = stateAsArray[1];
            float theta = stateAsArray[2];
            float theta_dot = stateAsArray[3];

            float force = action == 1 ? this.force_mag : -this.force_mag;
            float costheta = (float)Math.Cos(theta);
            float sintheta = (float)Math.Sin(theta);

            // For the interested reader:
            // https://coneural.org/florian/papers/05_cart_pole.pdf
            float temp = (force + this.polemass_length * (float)Math.Pow(theta_dot, 2) * sintheta) / this.total_mass;
            float thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0f / 3.0f - this.masspole * (float)Math.Pow(costheta, 2) / this.total_mass));
            float xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass;

            if (this.kinematics_integrator == "euler")
            {
                x = x + this.tau * x_dot;
                x_dot = x_dot + this.tau * xacc;
                theta = theta + this.tau * theta_dot;
                theta_dot = theta_dot + this.tau * thetaacc;
            }
            else
            { //:  // semi-implicit euler
                x_dot = x_dot + this.tau * xacc;
                x = x + this.tau * x_dot;
                theta_dot = theta_dot + this.tau * thetaacc;//
                theta = theta + this.tau * theta_dot;
            }
            this.state = new[] { x, x_dot, theta, theta_dot };

            var done = (
                x < -this.x_threshold
                || x > this.x_threshold
                || theta < -this.theta_threshold_radians
                || theta > this.theta_threshold_radians
            );

            var reward = 0f;
            if (!done)
            {
                reward = 1.0f;
            }
            else if (this.steps_beyond_done is null)
            {
                this.steps_beyond_done = 0;
                reward = 1.0f;
            }
            // Pole just fell!

            else
            {
                if (this.steps_beyond_done == 0)
                    logger.warn(
                        "You are calling 'step()' even though this " +
                        "environment has already returned done = True. You " +
                        "should always call 'reset()' once you receive 'done = " +
                        "True' -- any further steps are undefined behavior."
                );
                this.steps_beyond_done += 1;
            }

            

            //return (np.array(this.state), reward, done, new object[] { });
            var result = new EnvorinmentStepResult
            {
                state = np.array(this.state),
                reward = reward,
                done = done,
                data = new object[] { }
            };
            return result;
        }
    }
    public class EnvorinmentStepResult
    {
        public Tensor state;
        public float reward;
        public bool done;
        public object data;
    }

    public class spaces
    {
        public static Discrete Discrete(int value) => new Discrete(value);

        internal static Box Box(NDArray x, NDArray y, Type dtype)
            => new Box(x, y, dtype);

    }
    public class Discrete : spaces
    {
        public Discrete(int size)
        {
            this.Size = size;
        }

        public int Size { get; }
    }
    public class Box : spaces
    {
        public Box(NDArray x, NDArray y, Type dtype)
        {
            X = x;
            Y = y;
            Dtype = dtype;
        }

        public NDArray X { get; }
        public NDArray Y { get; }
        public Type Dtype { get; }
    }
}