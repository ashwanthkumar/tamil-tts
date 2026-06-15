import {Composition} from 'remotion';
import {Intro} from './Intro';
import script from './script.json';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Intro"
      component={Intro}
      durationInFrames={script.total}
      fps={script.fps}
      width={script.width}
      height={script.height}
    />
  );
};
